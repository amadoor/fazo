//! Follows V8's optimization test natives (`%PrepareFunctionForOptimization` and related).
//! Opens only with `-X jitlib`, and changes only the tier a function runs in, never its results.

use super::{arg_error, argument, got};
use crate::{LuaError, Tier, Values, Vm, arena::Ref, proto::Proto, value::Value, values};

pub(super) fn open(vm: &mut Vm) {
    let table = vm.new_table();
    let functions: &[(&'static str, crate::object::NativeFn)] = &[
        ("prepare", prepare),
        ("compile", compile),
        ("osr", osr),
        ("never", never),
        ("deoptimize", deoptimize),
        ("tier", tier),
    ];
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    vm.set_global("jit", Value::Table(table));
}

fn lua_function(vm: &mut Vm, args: &[Value], name: &str) -> Result<Ref<Proto>, LuaError> {
    if let Value::Closure(closure) = argument(args, 0)
        && let Some(proto) = vm.closure_proto(closure)
    {
        return Ok(proto);
    }
    let message = format!("Lua function expected, got {}", got(vm, args, 0));
    Err(arg_error(vm, 1, name, &message))
}

fn tier_name(vm: &mut Vm, tier: Tier) -> Value {
    vm.string_value(tier.name().as_bytes())
}

/// Compiles to Baseline at the next call or back edge, ignoring the budget and payoff estimate.
fn prepare(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "prepare")?;
    vm.jit_prepare(proto, false);
    Ok(values![])
}

fn compile(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "compile")?;
    let wanted = match argument(args, 1) {
        Value::Nil => Tier::Peak,
        value => match vm.string_bytes(value).and_then(Tier::from_name) {
            Some(tier) => tier,
            None => {
                let message = format!("invalid tier, got {}", got(vm, args, 1));
                return Err(arg_error(vm, 2, "compile", &message));
            }
        },
    };
    let reached = vm.jit_debug_compile(proto, wanted);
    Ok(values![tier_name(vm, reached)])
}

fn osr(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "osr")?;
    vm.jit_prepare(proto, true);
    Ok(values![])
}

fn never(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "never")?;
    vm.jit_never(proto);
    Ok(values![])
}

/// A running frame of `f` keeps its compiled code until its next exit.
fn deoptimize(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "deoptimize")?;
    vm.jit_deoptimize(proto);
    Ok(values![])
}

fn tier(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let proto = lua_function(vm, args, "tier")?;
    let tier = vm.jit_tier_of(proto);
    Ok(values![tier_name(vm, tier)])
}
