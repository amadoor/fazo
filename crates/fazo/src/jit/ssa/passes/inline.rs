//! Follows V8 Maglev's `IsFunctionCandidateForEagerInlining` and `ShouldEagerInlineCall` (src/maglev/maglev-graph-builder.cc).
//! An inlined frame cannot leave compiled code through a helper, so the builder also checks that its calls inline.

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
    let inlinable = proto.code.iter().enumerate().all(|(pc, &instruction)| inlinable(proto, pc, instruction));
    inlinable.then_some(Candidate { proto, reference })
}

/// `RETURN` with every result follows each `TAILCALL`, which never reaches it when the callee inlines.
fn inlinable(proto: &Proto, pc: usize, instruction: Instruction) -> bool {
    match instruction {
        Instruction::Jmp { sj } => sj >= 0,
        Instruction::Return { b: 0, k: false, .. } => pc > 0 && matches!(proto.code[pc - 1], Instruction::TailCall { .. }),
        Instruction::Return { k, .. } => !k,
        Instruction::TailCall { b, k, .. } => b != 0 && !k,
        Instruction::Call { b, c, .. } => b != 0 && c != 0,
        Instruction::NewTable { c, k, .. } => c == 0 && !k,
        Instruction::Closure { .. }
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
