pub(crate) mod address;
pub(crate) mod bounds;
pub(crate) mod dce;
pub(crate) mod escape;
pub(crate) mod fold;
pub(crate) mod forward;
pub(crate) mod hoist;
pub(crate) mod inline;
pub(crate) mod repr;
pub(crate) mod types;

#[cfg(test)]
mod tests;

use super::ir::{BlockId, Func, Op, Rep, Type, Val};

pub(crate) type Pass = (&'static str, fn(&mut Func));

pub(crate) const PEAK: &[Pass] = &[
    ("types", types::run),
    ("forward", forward::run),
    ("fold", fold::run),
    ("types", types::run),
    ("fold", fold::run),
    ("escape", escape::run),
    ("repr", repr::run),
    ("hoist", hoist::run),
    ("bounds", bounds::run),
    ("fold", fold::run),
    ("dce", dce::run),
];

/// Peak's pipeline without the last `fold`, whose compile was not repaid under `force=optimizing`. Keeping the first `types` avoids a boxed accumulator in spectral-norm under default tiering.
pub(crate) const OPTIMIZING: &[Pass] = &[
    ("types", types::run),
    ("forward", forward::run),
    ("fold", fold::run),
    ("types", types::run),
    ("fold", fold::run),
    ("escape", escape::run),
    ("repr", repr::run),
    ("hoist", hoist::run),
    ("bounds", bounds::run),
    ("dce", dce::run),
    ("address", address::run),
];

/// Follows V8's `--maglev-verify-graph`.
pub(crate) fn optimize(func: &mut Func, pipeline: &[Pass], verify: bool) -> Result<(), String> {
    let mut reps_selected = false;
    for &(name, pass) in pipeline {
        let before = verify.then(|| func.clone());
        pass(func);
        reps_selected |= name == "repr";
        if let Some(before) = before {
            super::check::check(func, reps_selected).map_err(|message| format!("after {name}: {message}
{before}"))?;
        }
    }
    Ok(())
}

pub(crate) fn insert(func: &mut Func, block: BlockId, position: usize, op: Op, ty: Type, rep: Rep) -> Val {
    let value = func.add_inst(op, ty, rep);
    let insts = &mut func.block_mut(block).insts;
    let position = position.min(insts.len());
    insts.insert(position, value);
    value
}

/// Uses must already be rewritten when these are removed.
pub(crate) fn remove(func: &mut Func, dead: &[bool]) {
    if !dead.contains(&true) {
        return;
    }
    for block in &mut func.blocks {
        block.insts.retain(|value| !dead.get(value.0 as usize).copied().unwrap_or(false));
    }
}
