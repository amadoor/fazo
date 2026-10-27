//! The Optimizing tier's one-pass backend, after V8's Maglev (src/maglev/maglev-regalloc.cc, src/maglev/maglev-code-generator.cc).

mod alloc;
mod asm;
mod calls;
mod codegen;
mod exits;
mod live;
mod nodes;
mod regs;
mod unwind;
#[cfg(any(target_arch = "aarch64", test))]
mod a64;
#[cfg(any(target_arch = "x86_64", test))]
mod x64;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use self::{asm::Asm, codegen::Gen, exits::ExitTable};
use super::{DeoptPoint, ir::{Func, Op}, lower::Env, runtime};
use crate::jit::{CodeBudget, JitCtx, baseline::BaselineUnit, sites};

#[cfg(target_arch = "x86_64")]
type Host = x64::X64;
#[cfg(target_arch = "aarch64")]
type Host = a64::A64;

pub(crate) struct Addresses {
    step: u64,
    fast: [u64; sites::FAST_NAMES.len()],
    value: u64,
    pure_float: u64,
    pure_int: u64,
    loop_count: u64,
    mixed: u64,
    new_table: u64,
    materialize: u64,
    safe_point: u64,
    tier_check: u64,
    resume: u64,
    exit: u64,
    store: u64,
}

impl Addresses {
    fn new() -> Self {
        let runtime: Vec<(&str, *const u8)> = runtime::symbols().to_vec();
        let find = |name: &str| runtime.iter().find(|(known, _)| *known == name).map_or(0, |&(_, address)| address as u64);
        let fast = sites::fast_symbols().map(|(_, address)| address as u64);
        Self {
            step: crate::jit::helpers::step as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            fast,
            value: find(runtime::VALUE),
            pure_float: find(runtime::PURE_FLOAT),
            pure_int: find(runtime::PURE_INT),
            loop_count: find(runtime::LOOP_COUNT),
            mixed: find(runtime::MIXED),
            new_table: find(runtime::NEW_TABLE),
            materialize: find(runtime::MATERIALIZE),
            safe_point: find(runtime::SAFE_POINT),
            tier_check: runtime::tier_check as extern "C" fn(*mut JitCtx, u32) -> i64 as usize as u64,
            resume: find(runtime::RESUME),
            exit: exits::exit as exits::ExitFn as usize as u64,
            store: exits::store as exits::ExitFn as usize as u64,
        }
    }
}

/// Mapped code with the exit table it addresses; the table outlives every run of the code.
pub(crate) struct Native {
    pub(crate) unit: BaselineUnit,
    pub(crate) _table: Box<ExitTable>,
    pub(crate) bytes: usize,
    /// The code as hexadecimal words, for `-X trace`.
    pub(crate) listing: Option<String>,
}

impl Native {
    pub(crate) fn entry(&self) -> *const u8 {
        self.unit.entry()
    }
}

fn materialize_values(func: &Func) -> usize {
    let largest = func.insts.iter().filter_map(|inst| match &inst.op {
        Op::Materialize { object } => Some(object.fields.len() + 1),
        _ => None,
    });
    largest.max().unwrap_or(0).max(3)
}

fn calls_directly(func: &Func) -> bool {
    func.insts.iter().any(|inst| matches!(inst.op, Op::CallLua { .. })) || func.blocks.iter().any(|block| matches!(block.term, super::ir::Term::TailCall { .. }))
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn compile(func: &Func, env: &Env<'_>, points: Box<[DeoptPoint]>, budget: Arc<CodeBudget>, listing: bool) -> Option<Native> {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("sse4.1") {
        return None;
    }
    let (code, trailer) = assemble::<Host>(func, env, points)?;
    let bytes = code.0.len();
    let unit = BaselineUnit::install(&code.0, trailer, budget)?;
    let listing = listing.then(|| code.0.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
    Some(Native { unit, _table: code.1, bytes, listing })
}

/// Charges the entry to the budget as Baseline does at a return, before any value lives in a register.
fn charge_entry<A: Asm>(asm: &mut A, cell: i64, (body, exit_with): (asm::Label, asm::Label), tier_check: u64) {
    let asm::ArgLoc::Reg(temp) = A::arg(2) else {
        unreachable!("the third argument goes in a register");
    };
    let asm::ArgLoc::Reg(context) = A::arg(0) else {
        unreachable!("the first argument goes in a register");
    };
    let asm::ArgLoc::Reg(pc) = A::arg(1) else {
        unreachable!("the second argument goes in a register");
    };
    asm.mov_imm(temp, cell);
    asm.sub_mem(temp, 0, crate::jit::ENTRY_CHARGE as i32);
    asm.branch(asm::Cond::Gt, body);
    asm.mov(context, A::CTX);
    asm.mov_imm(pc, 0);
    asm.call(tier_check);
    asm.mov(A::SCRATCH, A::result());
    asm.cmp_imm(A::SCRATCH, 0);
    asm.branch(asm::Cond::Eq, body);
    asm.jump(exit_with);
}

fn place_entry_arguments<A: Asm>(asm: &mut A) {
    let asm::ArgLoc::Reg(context) = A::arg(0) else {
        unreachable!("the first argument goes in a register");
    };
    let asm::ArgLoc::Reg(pc) = A::arg(1) else {
        unreachable!("the second argument goes in a register");
    };
    asm.mov(context, A::CTX);
    asm.mov_imm(pc, 0);
}

type Assembled = ((Vec<u8>, Box<ExitTable>), Option<crate::jit::unwind::Trailer>);

/// The body first, then the prologue, which is placed in front of it once the frame size is known.
fn assemble<A: Asm>(func: &Func, env: &Env<'_>, points: Box<[DeoptPoint]>) -> Option<Assembled> {
    let addresses = Addresses::new();
    let table = Box::new(ExitTable { states: func.states.clone(), points, ..ExitTable::default() });
    let scratch_at = A::OUTGOING;
    let scratch_size = materialize_values(func) as i32 * codegen::VALUE_SIZE;
    let ctx_at = scratch_at + scratch_size;
    let ctx_size = if calls_directly(func) { std::mem::size_of::<JitCtx>().next_multiple_of(16) as i32 } else { 0 };
    let spill_at = ctx_at + ctx_size;
    let generator = Gen::<A>::new(func, env, &addresses, table, (scratch_at, ctx_at, spill_at));
    let mut generated = generator.run()?;

    let callee_saved_floats: Vec<u8> = generated.used.iter().filter(|reg| reg.class == regs::Class::Fp && A::REGISTERS.callee_saved.contains(*reg)).map(|reg| reg.hw).collect();
    let float_at = (spill_at as u32 + 8 * generated.slots).next_multiple_of(16);
    let floats: Vec<(u8, u32)> = callee_saved_floats.iter().enumerate().map(|(index, &reg)| (reg, float_at + 16 * index as u32)).collect();
    let saved = A::saved_registers(generated.used);
    let frame = A::frame_size(float_at + 16 * floats.len() as u32, saved.len());
    if frame > A::MAX_FRAME {
        return None;
    }

    let asm = &mut generated.asm;
    asm.bind(generated.epilogue);
    asm.epilogue(frame, &floats, A::SCRATCH, &saved);
    asm.bind(generated.tail_exit);
    place_entry_arguments(asm);
    asm.tail_epilogue(frame, &floats, &saved);
    let dispatch = asm.offset();
    if !generated.falls_into {
        for &(pc, label) in &generated.roots {
            asm.cmp_imm(A::SCRATCH, pc as i32);
            match env.tier_budget {
                Some(cell) if pc == super::entry_pc(env.vararg) => {
                    let skip = asm.new_label();
                    asm.branch(asm::Cond::Ne, skip);
                    charge_entry(asm, cell as i64, (label, generated.exit_with), addresses.tier_check);
                    asm.bind(skip);
                }
                _ => asm.branch(asm::Cond::Eq, label),
            }
        }
        asm.trap();
    }
    let body = generated.asm.finish()?;

    let mut head = A::new();
    let prologue = head.prologue(frame, &floats, (!generated.falls_into).then_some(A::SCRATCH), &saved);
    let mut code = head.finish()?;
    if !generated.falls_into {
        code.extend(A::raw_jump(dispatch));
    }
    code.extend_from_slice(&body);

    let code_len = code.len();
    let trailer_offset = code_len.next_multiple_of(crate::jit::unwind::TRAILER_ALIGNMENT);
    let trailer = unwind::trailer(&prologue, code_len, trailer_offset);
    if let Some(trailer) = &trailer {
        code.resize(trailer_offset, 0);
        code.extend_from_slice(&trailer.bytes);
    }
    Some(((code, generated.table), trailer))
}

