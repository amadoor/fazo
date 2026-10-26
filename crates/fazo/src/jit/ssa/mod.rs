//! Follows V8 Maglev (src/maglev/).

pub(crate) mod build;
pub(crate) mod check;
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

/// The points outlive the code, which addresses them.
pub(crate) struct PeakUnit {
    pub(crate) unit: super::JitUnit,
    pub(crate) _points: Box<[DeoptPoint]>,
    pub(crate) entries: Vec<u32>,
    pub(crate) shapes: Vec<crate::arena::Ref<crate::shape::Shape>>,
}

impl PeakUnit {
    pub(crate) fn entry(&self) -> *const u8 {
        self.unit.entry()
    }

    pub(crate) fn enters_at(&self, pc: usize) -> bool {
        self.entries.iter().any(|&entry| entry as usize == pc)
    }
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
