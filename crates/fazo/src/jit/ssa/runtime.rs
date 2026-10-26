//! Value operations answer zero when a metamethod, coercion or error is needed, so the frame deoptimizes.

use super::{DeoptPoint, ir::{BuiltinFn, NumOp}};
use crate::{
    instruction::{ArithOp, UnaryOp},
    jit::JitCtx,
    number,
    value::Value,
    vm::fast,
};

pub(crate) const DEOPT: &str = "fazo_ssa_deopt";
pub(crate) const SAFE_POINT: &str = "fazo_ssa_safe_point";
pub(crate) const RESUME: &str = "fazo_ssa_resume";
pub(crate) const VALUE: &str = "fazo_ssa_value";
pub(crate) const PURE_FLOAT: &str = "fazo_ssa_pure_float";
pub(crate) const PURE_INT: &str = "fazo_ssa_pure_int";
pub(crate) const LOOP_COUNT: &str = "fazo_ssa_loop_count";
pub(crate) const MIXED: &str = "fazo_ssa_mixed";
pub(crate) const NEW_TABLE: &str = "fazo_ssa_new_table";
pub(crate) const MATERIALIZE: &str = "fazo_ssa_materialize";
pub(crate) const RAN_CODE: i64 = 1 << 62;

pub(crate) const KIND_ARITH: u32 = 0;
pub(crate) const KIND_COMPARE: u32 = 1;
pub(crate) const KIND_INDEX: u32 = 2;
pub(crate) const KIND_NEW_INDEX: u32 = 3;
pub(crate) const KIND_LEN: u32 = 4;
pub(crate) const KIND_SET_UPVALUE: u32 = 5;
pub(crate) const KIND_BARRIER: u32 = 6;
pub(crate) const KIND_BUILTIN: u32 = 7;

pub(crate) const PURE_MOD: i32 = 0;
pub(crate) const PURE_POW: i32 = 1;
pub(crate) const PURE_SHL: i32 = 0;
pub(crate) const PURE_SHR: i32 = 1;

pub(crate) const MIXED_LT_IF: i32 = 0;
pub(crate) const MIXED_LE_IF: i32 = 1;
pub(crate) const MIXED_LT_FI: i32 = 2;
pub(crate) const MIXED_LE_FI: i32 = 3;
pub(crate) const MIXED_EQ_FI: i32 = 4;

const NUM_OPS: [NumOp; 14] = [
    NumOp::Add,
    NumOp::Sub,
    NumOp::Mul,
    NumOp::Div,
    NumOp::Mod,
    NumOp::IDiv,
    NumOp::Pow,
    NumOp::BAnd,
    NumOp::BOr,
    NumOp::BXor,
    NumOp::Shl,
    NumOp::Shr,
    NumOp::Neg,
    NumOp::BNot,
];

pub(crate) fn num_op_code(op: NumOp) -> u32 {
    NUM_OPS.iter().position(|&known| known == op).unwrap_or(0) as u32
}

type DeoptFn = extern "C" fn(*mut JitCtx, i64) -> i64;
type SafePointFn = extern "C" fn(*mut JitCtx, u32) -> i64;
type ValueFn = extern "C" fn(*mut JitCtx, u32, *mut Value, u32) -> i64;
type PureFloatFn = extern "C" fn(i32, f64, f64) -> f64;
type PureIntFn = extern "C" fn(i32, i64, i64) -> i64;
type LoopCountFn = extern "C" fn(i64, i64, i64) -> i64;
type MixedFn = extern "C" fn(i32, i64, f64) -> i64;
type NewTableFn = extern "C" fn(*mut JitCtx, u32, u32, u32) -> i64;
type MaterializeFn = extern "C" fn(*mut JitCtx, u64, u32, *const Value, u32) -> i64;

pub(crate) fn symbols() -> [(&'static str, *const u8); 10] {
    [
        (DEOPT, deopt as DeoptFn as *const u8),
        (SAFE_POINT, safe_point as SafePointFn as *const u8),
        (RESUME, resume as DeoptFn as *const u8),
        (VALUE, value as ValueFn as *const u8),
        (PURE_FLOAT, pure_float as PureFloatFn as *const u8),
        (PURE_INT, pure_int as PureIntFn as *const u8),
        (LOOP_COUNT, loop_count as LoopCountFn as *const u8),
        (MIXED, mixed as MixedFn as *const u8),
        (NEW_TABLE, new_table as NewTableFn as *const u8),
        (MATERIALIZE, materialize as MaterializeFn as *const u8),
    ]
}

extern "C" fn materialize(ctx: *mut JitCtx, shape: u64, hash_log: u32, values: *const Value, count: u32) -> i64 {
    // SAFETY: compiled code passes a stack buffer it filled with `count` values.
    let values = unsafe { std::slice::from_raw_parts(values, count as usize) };
    let shape = crate::arena::Ref::new(shape as u32, (shape >> 32) as u32);
    with_vm(ctx, |ctx, vm| {
        let table = vm.jit_materialize(shape, hash_log as u8, values[0], &values[1..]);
        vm.jit_refresh_context(ctx);
        table.packed() as i64
    })
}

extern "C" fn new_table(ctx: *mut JitCtx, pc: u32, hash_log: u32, top: u32) -> i64 {
    with_vm(ctx, |ctx, vm| {
        let Some((table, ran)) = vm.jit_alloc_table(ctx.base, (pc as usize, top as usize), hash_log as u8) else {
            return -1;
        };
        vm.jit_refresh_context(ctx);
        let packed = table.packed() as i64;
        if ran { packed | RAN_CODE } else { packed }
    })
}

fn with_vm<T>(ctx: *mut JitCtx, body: impl FnOnce(&mut JitCtx, &mut crate::Vm) -> T) -> T {
    // SAFETY: compiled code passes the context the driver built for this frame, which outlives the call, and its VM
    // pointer names the VM running the code on this thread.
    let ctx = unsafe { &mut *ctx };
    // SAFETY: as above; nothing else holds the VM while compiled code runs.
    let vm = unsafe { &mut *ctx.vm };
    body(ctx, vm)
}

extern "C" fn deopt(ctx: *mut JitCtx, point: i64) -> i64 {
    // SAFETY: `point` is the address of one of the deoptimization points the unit owns, and the unit stays mapped,
    // with its points, while any frame runs its code.
    let point = unsafe { &*(point as usize as *const DeoptPoint) };
    with_vm(ctx, |ctx, vm| vm.jit_deopt(ctx, point)).wrapping_neg()
}

extern "C" fn safe_point(ctx: *mut JitCtx, target: u32) -> i64 {
    with_vm(ctx, |ctx, vm| {
        if vm.jit_exit_stressed(target as usize) {
            return -crate::jit::EXIT_HOOK;
        }
        vm.jit_safe_point(ctx, target as usize)
    })
}

/// Optimizing code out of budget asks whether Peak pays now.
pub(crate) extern "C" fn tier_check(ctx: *mut JitCtx, pc: u32) -> i64 {
    with_vm(ctx, |ctx, vm| vm.jit_tier_check(ctx, pc as usize, crate::Tier::Optimizing))
}

extern "C" fn resume(ctx: *mut JitCtx, pc: i64) -> i64 {
    with_vm(ctx, |_, vm| {
        vm.save_pc(pc as usize);
        -crate::jit::EXIT_RELOAD
    })
}

extern "C" fn value(ctx: *mut JitCtx, kind: u32, operands: *mut Value, extra: u32) -> i64 {
    // SAFETY: compiled code passes its scratch slot of three values, every one it reads written before the call.
    let operands = unsafe { std::slice::from_raw_parts_mut(operands, 3) };
    with_vm(ctx, |ctx, vm| match kind {
        KIND_ARITH => store(operands, arith(extra, operands[0], operands[1])),
        KIND_COMPARE => match compare(extra, operands[0], operands[1]) {
            Some(true) => 2,
            Some(false) => 1,
            None => 0,
        },
        KIND_INDEX => store(operands, vm.index_resolved(operands[0], operands[1])),
        KIND_NEW_INDEX => i64::from(vm.jit_raw_set(operands[0], operands[1], operands[2])),
        KIND_LEN => store(operands, vm.raw_length(operands[0])),
        KIND_SET_UPVALUE => {
            if let Value::Closure(closure) = operands[1] {
                vm.jit_set_closure_upvalue(closure, extra as u8, operands[0]);
            }
            1
        }
        KIND_BUILTIN => match builtin_of(extra) {
            Some(builtin) => store(operands, vm.jit_builtin(builtin, operands[0], operands[1])),
            None => 0,
        },
        KIND_BARRIER => {
            if let Value::Table(table) = operands[0] {
                vm.jit_table_barrier(table, operands[1]);
            }
            let _ = ctx;
            1
        }
        _ => 0,
    })
}

const BUILTINS: [BuiltinFn; 5] = [BuiltinFn::Type, BuiltinFn::RawEqual, BuiltinFn::RawGet, BuiltinFn::GetMetatable, BuiltinFn::SetMetatable];

pub(crate) fn builtin_code(function: BuiltinFn) -> u32 {
    BUILTINS.iter().position(|&known| known == function).unwrap_or(0) as u32
}

fn builtin_of(code: u32) -> Option<crate::vm::Builtin> {
    Some(match BUILTINS.get(code as usize)? {
        BuiltinFn::Type => crate::vm::Builtin::Type,
        BuiltinFn::RawEqual => crate::vm::Builtin::RawEqual,
        BuiltinFn::RawGet => crate::vm::Builtin::RawGet,
        BuiltinFn::GetMetatable => crate::vm::Builtin::GetMetatable,
        BuiltinFn::SetMetatable => crate::vm::Builtin::SetMetatable,
    })
}

fn store(operands: &mut [Value], result: Option<Value>) -> i64 {
    let Some(result) = result else {
        return 0;
    };
    operands[0] = result;
    1
}

fn arith(code: u32, left: Value, right: Value) -> Option<Value> {
    let op = NUM_OPS.get(code as usize).copied()?;
    let binary = match op {
        NumOp::Neg => return fast::unary(UnaryOp::Minus, left),
        NumOp::BNot => return fast::unary(UnaryOp::BNot, left),
        NumOp::Add => ArithOp::Add,
        NumOp::Sub => ArithOp::Sub,
        NumOp::Mul => ArithOp::Mul,
        NumOp::Div => ArithOp::Div,
        NumOp::Mod => ArithOp::Mod,
        NumOp::IDiv => ArithOp::IDiv,
        NumOp::Pow => ArithOp::Pow,
        NumOp::BAnd => ArithOp::BAnd,
        NumOp::BOr => ArithOp::BOr,
        NumOp::BXor => ArithOp::BXor,
        NumOp::Shl => ArithOp::Shl,
        NumOp::Shr => ArithOp::Shr,
    };
    fast::arith(binary, left, right)
}

fn compare(cc: u32, left: Value, right: Value) -> Option<bool> {
    match cc {
        0 => fast::equal(left, right),
        1 => fast::less_than(left, right),
        _ => fast::less_equal(left, right),
    }
}

extern "C" fn pure_float(kind: i32, a: f64, b: f64) -> f64 {
    let op = if kind == PURE_MOD { ArithOp::Mod } else { ArithOp::Pow };
    number::float_arith(op, a, b)
}

extern "C" fn pure_int(kind: i32, a: i64, b: i64) -> i64 {
    if kind == PURE_SHL { number::shift_left(a, b) } else { number::shift_left(a, b.wrapping_neg()) }
}

extern "C" fn loop_count(start: i64, limit: i64, step: i64) -> i64 {
    crate::vm::int_loop_count(start, limit, step) as i64
}

extern "C" fn mixed(kind: i32, integer: i64, float: f64) -> i64 {
    let result = match kind {
        MIXED_LT_IF => number::less_integer_float(integer, float),
        MIXED_LE_IF => number::less_equal_integer_float(integer, float),
        MIXED_LT_FI => number::less_float_integer(float, integer),
        MIXED_LE_FI => number::less_equal_float_integer(float, integer),
        _ => number::float_equals_integer(float, integer),
    };
    i64::from(result)
}
