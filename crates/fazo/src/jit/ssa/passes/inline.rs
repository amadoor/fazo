//! Follows V8 Maglev's `IsFunctionCandidateForEagerInlining` and `ShouldEagerInlineCall` (src/maglev/maglev-graph-builder.cc).
//! Calls out are excluded so a failed guard can rebuild the inlined frame as an interpreter frame.

use crate::{arena::Ref, instruction::Instruction, proto::Proto};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    pub(crate) max_instructions: usize,
    pub(crate) depth: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self { max_instructions: 40, depth: 1 }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Candidate<'a> {
    pub(crate) proto: &'a Proto,
    pub(crate) reference: Ref<Proto>,
}

pub(crate) fn candidate<'a>(proto: &'a Proto, reference: Ref<Proto>, policy: &Policy) -> Option<Candidate<'a>> {
    if proto.is_vararg || proto.code.len() > policy.max_instructions {
        return None;
    }
    let inlinable = proto.code.iter().all(|&instruction| inlinable(instruction));
    inlinable.then_some(Candidate { proto, reference })
}

fn inlinable(instruction: Instruction) -> bool {
    match instruction {
        Instruction::Jmp { sj } => sj >= 0,
        Instruction::Return { b, k, .. } => b != 0 && !k,
        Instruction::Call { .. }
        | Instruction::TailCall { .. }
        | Instruction::Closure { .. }
        | Instruction::NewTable { .. }
        | Instruction::Concat { .. }
        | Instruction::Vararg { .. }
        | Instruction::VarargPrep { .. }
        | Instruction::GetVarg { .. }
        | Instruction::ErrNil { .. }
        | Instruction::Close { .. }
        | Instruction::Tbc { .. }
        | Instruction::TForPrep { .. }
        | Instruction::TForCall { .. }
        | Instruction::TForLoop { .. }
        | Instruction::ForPrep { .. }
        | Instruction::ForLoop { .. }
        | Instruction::SetList { .. }
        | Instruction::LoadKx { .. } => false,
        _ => true,
    }
}
