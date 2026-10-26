//! Follows V8 Maglev's `BytecodeAnalysis` (src/maglev/maglev-graph-builder.cc).

use crate::{instruction::Instruction, proto::Proto};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RegSet([u64; 4]);

impl RegSet {
    pub(crate) fn insert(&mut self, register: usize) {
        if register < 256 {
            self.0[register / 64] |= 1 << (register % 64);
        }
    }

    pub(crate) fn remove(&mut self, register: usize) {
        if register < 256 {
            self.0[register / 64] &= !(1 << (register % 64));
        }
    }

    pub(crate) fn contains(&self, register: usize) -> bool {
        register < 256 && self.0[register / 64] & (1 << (register % 64)) != 0
    }

    fn union_with(&mut self, other: &RegSet) -> bool {
        let mut changed = false;
        for (word, other) in self.0.iter_mut().zip(other.0) {
            let merged = *word | other;
            changed |= merged != *word;
            *word = merged;
        }
        changed
    }

    fn range(&mut self, from: usize, to: usize) {
        for register in from..to {
            self.insert(register);
        }
    }
}

pub(crate) struct Flow {
    pub(crate) successors: Vec<Vec<usize>>,
    pub(crate) leaders: Vec<bool>,
    pub(crate) back_edges: Vec<bool>,
    pub(crate) headers: Vec<bool>,
    pub(crate) live: Vec<RegSet>,
}

struct Effects {
    uses: RegSet,
    kills: RegSet,
}

pub(crate) fn analyze(proto: &Proto, captured: &RegSet) -> Flow {
    let length = proto.code.len();
    let successors: Vec<Vec<usize>> = (0..length).map(|pc| successors(proto, pc)).collect();
    let mut leaders = vec![false; length];
    let mut back_edges = vec![false; length];
    let mut headers = vec![false; length];
    if length > 0 {
        leaders[0] = true;
    }
    for (pc, targets) in successors.iter().enumerate() {
        mark_leaders(proto, pc, targets, &mut leaders);
        for &target in targets {
            if target <= pc {
                back_edges[pc] = true;
                headers[target] = true;
            }
        }
    }
    let live = liveness(proto, &successors, captured);
    Flow { successors, leaders, back_edges, headers, live }
}

fn mark_leaders(proto: &Proto, pc: usize, targets: &[usize], leaders: &mut [bool]) {
    let falls_through = targets.len() == 1 && targets[0] == pc + width(proto, pc);
    for &target in targets {
        if target < leaders.len() && !(falls_through && target == pc + width(proto, pc)) {
            leaders[target] = true;
        }
    }
    if !falls_through && pc + 1 < leaders.len() {
        leaders[pc + 1] = true;
    }
}

/// Some ops consume the following instruction: an `ExtraArg` or the arithmetic's `MmBin` fallback.
pub(crate) fn width(proto: &Proto, pc: usize) -> usize {
    match proto.code[pc] {
        Instruction::LoadKx { .. } | Instruction::NewTable { .. } => 2,
        Instruction::SetList { k: true, .. } => 2,
        instruction if is_arithmetic(instruction) => 2,
        _ => 1,
    }
}

pub(crate) fn is_arithmetic(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::AddI { .. }
            | Instruction::AddK { .. }
            | Instruction::SubK { .. }
            | Instruction::MulK { .. }
            | Instruction::ModK { .. }
            | Instruction::PowK { .. }
            | Instruction::DivK { .. }
            | Instruction::IDivK { .. }
            | Instruction::BAndK { .. }
            | Instruction::BOrK { .. }
            | Instruction::BXorK { .. }
            | Instruction::ShrI { .. }
            | Instruction::ShlI { .. }
            | Instruction::Add { .. }
            | Instruction::Sub { .. }
            | Instruction::Mul { .. }
            | Instruction::Mod { .. }
            | Instruction::Pow { .. }
            | Instruction::Div { .. }
            | Instruction::IDiv { .. }
            | Instruction::BAnd { .. }
            | Instruction::BOr { .. }
            | Instruction::BXor { .. }
            | Instruction::Shl { .. }
            | Instruction::Shr { .. }
    )
}

fn offset(from: usize, by: i32) -> usize {
    (from as i64 + i64::from(by)).max(0) as usize
}

/// A test is followed by the `JMP` it takes when the condition holds.
pub(crate) fn test_target(proto: &Proto, pc: usize) -> Option<usize> {
    match proto.code.get(pc + 1)? {
        Instruction::Jmp { sj } => Some(offset(pc + 2, *sj)),
        _ => None,
    }
}

pub(crate) fn successors(proto: &Proto, pc: usize) -> Vec<usize> {
    let instruction = proto.code[pc];
    let length = proto.code.len();
    let next = pc + width(proto, pc);
    let targets = match instruction {
        Instruction::Jmp { sj } => vec![offset(pc + 1, sj)],
        instruction if instruction.is_test() => match test_target(proto, pc) {
            Some(target) => vec![pc + 2, target],
            None => vec![pc + 1],
        },
        Instruction::LFalseSkip { .. } => vec![pc + 2],
        Instruction::ForPrep { bx, .. } => vec![pc + 1, pc + bx as usize + 2],
        Instruction::ForLoop { bx, .. } | Instruction::TForLoop { bx, .. } => vec![(pc + 1).saturating_sub(bx as usize), pc + 1],
        Instruction::TForPrep { bx, .. } => vec![pc + 1 + bx as usize],
        Instruction::Return { .. } | Instruction::Return0 { .. } | Instruction::Return1 { .. } | Instruction::TailCall { .. } => Vec::new(),
        _ => vec![next],
    };
    targets.into_iter().filter(|&target| target < length).collect()
}

fn liveness(proto: &Proto, successors: &[Vec<usize>], captured: &RegSet) -> Vec<RegSet> {
    let length = proto.code.len();
    let top = usize::from(proto.max_stack).max(1);
    let effects: Vec<Effects> = (0..length).map(|pc| effects(proto, pc, top, captured)).collect();
    let mut live = vec![RegSet::default(); length];
    let mut changed = true;
    while changed {
        changed = false;
        for pc in (0..length).rev() {
            let mut out = RegSet::default();
            for &successor in &successors[pc] {
                out.union_with(&live[successor]);
            }
            for register in 0..top {
                if effects[pc].kills.contains(register) {
                    out.remove(register);
                }
            }
            out.union_with(&effects[pc].uses);
            if out != live[pc] {
                live[pc] = out;
                changed = true;
            }
        }
    }
    for (pc, set) in live.iter_mut().enumerate() {
        for register in 0..active_locals(proto, pc) {
            set.insert(register);
        }
        set.union_with(captured);
    }
    live
}

/// Named locals in scope are visible to `debug.getlocal`, so they stay live while in scope.
pub(crate) fn active_locals(proto: &Proto, pc: usize) -> usize {
    proto.local_vars.iter().filter(|local| local.start_pc <= pc && pc < local.end_pc).count()
}

fn effects(proto: &Proto, pc: usize, top: usize, captured: &RegSet) -> Effects {
    let mut uses = RegSet::default();
    let mut kills = RegSet::default();
    let a = |a: u8| usize::from(a);
    let open = |from: usize, count: u8, adjust: usize| if count == 0 { top } else { from + usize::from(count) - adjust };
    match proto.code[pc] {
        Instruction::Move { a: ra, b } => {
            uses.insert(a(b));
            kills.insert(a(ra));
        }
        Instruction::LoadI { a: ra, .. }
        | Instruction::LoadF { a: ra, .. }
        | Instruction::LoadK { a: ra, .. }
        | Instruction::LoadKx { a: ra }
        | Instruction::LoadFalse { a: ra }
        | Instruction::LFalseSkip { a: ra }
        | Instruction::LoadTrue { a: ra }
        | Instruction::GetUpval { a: ra, .. }
        | Instruction::GetTabUp { a: ra, .. }
        | Instruction::NewTable { a: ra, .. }
        | Instruction::Closure { a: ra, .. } => {
            kills.insert(a(ra));
            if matches!(proto.code[pc], Instruction::Closure { .. }) {
                uses.union_with(captured);
            }
        }
        Instruction::LoadNil { a: ra, b } => kills.range(a(ra), a(ra) + a(b) + 1),
        Instruction::SetUpval { a: ra, .. } | Instruction::Tbc { a: ra } | Instruction::ErrNil { a: ra, .. } | Instruction::Test { a: ra, .. } => uses.insert(a(ra)),
        Instruction::GetTable { a: ra, b, c } | Instruction::GetVarg { a: ra, b, c } => {
            uses.insert(a(b));
            uses.insert(a(c));
            kills.insert(a(ra));
        }
        Instruction::GetI { a: ra, b, .. } | Instruction::GetField { a: ra, b, .. } | Instruction::Unm { a: ra, b } | Instruction::BNot { a: ra, b } | Instruction::Not { a: ra, b } | Instruction::Len { a: ra, b } => {
            uses.insert(a(b));
            kills.insert(a(ra));
        }
        Instruction::SetTabUp { c, k, .. } => {
            if !k {
                uses.insert(a(c));
            }
        }
        Instruction::SetTable { a: ra, b, c, k } => {
            uses.insert(a(ra));
            uses.insert(a(b));
            if !k {
                uses.insert(a(c));
            }
        }
        Instruction::SetI { a: ra, c, k, .. } | Instruction::SetField { a: ra, c, k, .. } => {
            uses.insert(a(ra));
            if !k {
                uses.insert(a(c));
            }
        }
        Instruction::SelfOp { a: ra, b, .. } => {
            uses.insert(a(b));
            kills.insert(a(ra));
            kills.insert(a(ra) + 1);
        }
        Instruction::AddI { a: ra, b, .. }
        | Instruction::AddK { a: ra, b, .. }
        | Instruction::SubK { a: ra, b, .. }
        | Instruction::MulK { a: ra, b, .. }
        | Instruction::ModK { a: ra, b, .. }
        | Instruction::PowK { a: ra, b, .. }
        | Instruction::DivK { a: ra, b, .. }
        | Instruction::IDivK { a: ra, b, .. }
        | Instruction::BAndK { a: ra, b, .. }
        | Instruction::BOrK { a: ra, b, .. }
        | Instruction::BXorK { a: ra, b, .. }
        | Instruction::ShrI { a: ra, b, .. }
        | Instruction::ShlI { a: ra, b, .. } => {
            uses.insert(a(b));
            kills.insert(a(ra));
        }
        Instruction::Add { a: ra, b, c }
        | Instruction::Sub { a: ra, b, c }
        | Instruction::Mul { a: ra, b, c }
        | Instruction::Mod { a: ra, b, c }
        | Instruction::Pow { a: ra, b, c }
        | Instruction::Div { a: ra, b, c }
        | Instruction::IDiv { a: ra, b, c }
        | Instruction::BAnd { a: ra, b, c }
        | Instruction::BOr { a: ra, b, c }
        | Instruction::BXor { a: ra, b, c }
        | Instruction::Shl { a: ra, b, c }
        | Instruction::Shr { a: ra, b, c } => {
            uses.insert(a(b));
            uses.insert(a(c));
            kills.insert(a(ra));
        }
        Instruction::MmBin { a: ra, b, .. } => {
            uses.insert(a(ra));
            uses.insert(a(b));
        }
        Instruction::MmBinI { a: ra, .. } | Instruction::MmBinK { a: ra, .. } => uses.insert(a(ra)),
        Instruction::Concat { a: ra, b } => {
            uses.range(a(ra), a(ra) + a(b));
            kills.insert(a(ra));
        }
        Instruction::Eq { a: ra, b, .. } | Instruction::Lt { a: ra, b, .. } | Instruction::Le { a: ra, b, .. } => {
            uses.insert(a(ra));
            uses.insert(a(b));
        }
        Instruction::EqK { a: ra, .. } | Instruction::EqI { a: ra, .. } | Instruction::LtI { a: ra, .. } | Instruction::LeI { a: ra, .. } | Instruction::GtI { a: ra, .. } | Instruction::GeI { a: ra, .. } => {
            uses.insert(a(ra));
        }
        Instruction::TestSet { b, .. } => uses.insert(a(b)),
        Instruction::Call { a: ra, b, c } => {
            uses.range(a(ra), open(a(ra), b, 0));
            kills.range(a(ra), if c == 0 { top } else { a(ra) + a(c) - 1 });
        }
        Instruction::TailCall { a: ra, b, .. } => uses.range(a(ra), open(a(ra), b, 0)),
        Instruction::Return { a: ra, b, .. } => uses.range(a(ra), open(a(ra), b, 1)),
        Instruction::Return1 { a: ra } => uses.insert(a(ra)),
        Instruction::ForLoop { a: ra, .. } | Instruction::ForPrep { a: ra, .. } => uses.range(a(ra), a(ra) + 3),
        Instruction::TForPrep { a: ra, .. } | Instruction::TForCall { a: ra, .. } => uses.range(a(ra), a(ra) + 4),
        Instruction::TForLoop { a: ra, .. } => uses.range(a(ra), a(ra) + 5),
        Instruction::SetList { a: ra, b, .. } => uses.range(a(ra), open(a(ra), b, 0) + 1),
        Instruction::Vararg { a: ra, c, .. } => kills.range(a(ra), if c == 0 { top } else { a(ra) + a(c) - 1 }),
        Instruction::Return0 { .. } | Instruction::Close { .. } | Instruction::Jmp { .. } | Instruction::VarargPrep { .. } | Instruction::ExtraArg { .. } => {}
    }
    Effects { uses, kills }
}

/// Captured registers stay live, since open upvalues read and write them.
pub(crate) fn captured_registers(proto: &Proto, child: &dyn Fn(usize) -> Option<Vec<(bool, u8)>>) -> RegSet {
    let mut captured = RegSet::default();
    for instruction in &proto.code {
        let Instruction::Closure { bx, .. } = *instruction else {
            continue;
        };
        let Some(upvalues) = child(bx as usize) else {
            for register in 0..256 {
                captured.insert(register);
            }
            return captured;
        };
        for (in_stack, index) in upvalues {
            if in_stack {
                captured.insert(usize::from(index));
            }
        }
    }
    captured
}
