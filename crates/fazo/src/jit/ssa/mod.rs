//! Follows V8 Maglev (src/maglev/).

pub(crate) mod build;
pub(crate) mod check;
pub(crate) mod emit;
pub(crate) mod flow;
pub(crate) mod ir;
pub(crate) mod lower;
pub(crate) mod passes;
pub(crate) mod print;
pub(crate) mod runtime;

use ir::InlineFrame;

/// Follows V8's `DeoptimizationData`, one per exit.
pub(crate) struct DeoptPoint {
    pub(crate) pc: u32,
    pub(crate) site: u32,
    pub(crate) frames: Box<[InlineFrame]>,
    pub(crate) lazy: bool,
}

pub(crate) enum Code {
    Cranelift { unit: Box<super::JitUnit>, _points: Box<[DeoptPoint]> },
    Native(emit::Native),
}

/// Optimized code of either tier; the points and exits outlive the code, which addresses them.
pub(crate) struct PeakUnit {
    pub(crate) code: Code,
    pub(crate) entries: Vec<u32>,
    pub(crate) shapes: Vec<crate::arena::Ref<crate::shape::Shape>>,
}

impl PeakUnit {
    pub(crate) fn entry(&self) -> *const u8 {
        match &self.code {
            Code::Cranelift { unit, .. } => unit.entry(),
            Code::Native(native) => native.entry(),
        }
    }

    pub(crate) fn tier(&self) -> crate::Tier {
        match self.code {
            Code::Cranelift { .. } => crate::Tier::Peak,
            Code::Native(_) => crate::Tier::Optimizing,
        }
    }

    pub(crate) fn size(&self) -> usize {
        match &self.code {
            Code::Cranelift { unit, .. } => unit.ir_insts(),
            Code::Native(native) => native.bytes,
        }
    }

    pub(crate) fn enters_at(&self, pc: usize) -> bool {
        self.entries.iter().any(|&entry| entry as usize == pc)
    }
}

/// The driver runs VARARGPREP first, so vararg code starts at the next instruction.
pub(crate) fn entry_pc(vararg: bool) -> u32 {
    u32::from(vararg)
}

pub(crate) fn points(func: &ir::Func) -> Box<[DeoptPoint]> {
    func.states
        .iter()
        .map(|state| DeoptPoint {
            pc: state.pc,
            site: state.site,
            frames: state.frames.clone().into_boxed_slice(),
            lazy: state.lazy,
        })
        .collect()
}
