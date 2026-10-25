
use super::{
    arg_error, argument, check_any, check_integer, check_number,
    machine::{Input, Machine, Next, Op, run},
};
use crate::{
    LuaError, Values, values, Vm,
    number::{self, Rounding},
    object::{NativeFn, Step},
    value::Value,
    vm::Builtin,
};

const DEFAULT_SEED: i64 = 0;

pub(super) fn open(vm: &mut Vm) {
    let functions: &[(&'static str, NativeFn)] = &[
        ("type", math_type),
        ("tointeger", tointeger),
        ("floor", floor),
        ("ceil", ceil),
        ("abs", abs),
        ("fmod", fmod),
        ("modf", modf),
        ("ult", ult),
        ("sqrt", sqrt),
        ("frexp", frexp),
        ("ldexp", ldexp),
        ("exp", exp),
        ("sin", sin),
        ("cos", cos),
        ("tan", tan),
        ("asin", asin),
        ("acos", acos),
        ("atan", atan),
        ("log", log),
        ("deg", deg),
        ("rad", rad),
        ("random", random),
        ("randomseed", randomseed),
    ];
    let table = vm.new_table();
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        if let Some(builtin) = builtin(name) {
            vm.mark_builtin(value, builtin);
        }
        vm.set_field(table, name, value);
    }
    let maximum = vm.new_stepped("max", maximum);
    vm.mark_builtin(maximum, Builtin::Max);
    vm.set_field(table, "max", maximum);
    let minimum = vm.new_stepped("min", minimum);
    vm.mark_builtin(minimum, Builtin::Min);
    vm.set_field(table, "min", minimum);
    vm.set_field(table, "maxinteger", Value::Integer(i64::MAX));
    vm.set_field(table, "mininteger", Value::Integer(i64::MIN));
    vm.set_field(table, "huge", Value::Float(f64::INFINITY));
    vm.set_field(table, "pi", Value::Float(std::f64::consts::PI));
    vm.set_global("math", Value::Table(table));
    let seed = vm.options().random_seed.unwrap_or(DEFAULT_SEED);
    set_seed(vm, seed as u64, 0);
}

fn builtin(name: &str) -> Option<Builtin> {
    match name {
        "floor" => Some(Builtin::Floor),
        "ceil" => Some(Builtin::Ceil),
        "abs" => Some(Builtin::Abs),
        "sqrt" => Some(Builtin::Sqrt),
        _ => None,
    }
}

fn math_type(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    check_any(vm, args, 0, "type")?;
    let kind = match args[0] {
        Value::Integer(_) => vm.string_value(b"integer"),
        Value::Float(_) => vm.string_value(b"float"),
        _ => Value::Nil,
    };
    Ok(values![kind])
}

fn tointeger(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let converted = match argument(args, 0) {
        Value::Integer(integer) => Some(integer),
        Value::Float(float) => number::float_to_integer(float, Rounding::Exact),
        Value::String(string) => number::parse_number(vm.heap_bytes(string))
            .and_then(|number| number::to_integer(number, Rounding::Exact)),
        _ => None,
    };
    if converted.is_none() {
        check_any(vm, args, 0, "tointeger")?;
    }
    Ok(values![converted.map_or(Value::Nil, Value::Integer)])
}

fn push_number_int(rounded: f64) -> Value {
    match number::float_to_integer(rounded, Rounding::Exact) {
        Some(integer) => Value::Integer(integer),
        None => Value::Float(rounded),
    }
}

fn rounded(vm: &mut Vm, args: &[Value], name: &str, round: fn(f64) -> f64) -> Result<Values, LuaError> {
    if let Value::Integer(integer) = argument(args, 0) {
        return Ok(values![Value::Integer(integer)]);
    }
    let float = check_number(vm, args, 0, name)?.to_float();
    Ok(values![push_number_int(round(float))])
}

fn floor(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    rounded(vm, args, "floor", f64::floor)
}

fn ceil(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    rounded(vm, args, "ceil", f64::ceil)
}

fn abs(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if let Value::Integer(integer) = argument(args, 0) {
        return Ok(values![Value::Integer(integer.wrapping_abs())]);
    }
    let float = check_number(vm, args, 0, "abs")?.to_float();
    Ok(values![Value::Float(float.abs())])
}

fn ult(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let first = check_integer(vm, args, 0, "ult")?;
    let second = check_integer(vm, args, 1, "ult")?;
    Ok(values![Value::Boolean((first as u64) < (second as u64))])
}

fn fmod(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if let (Value::Integer(dividend), Value::Integer(divisor)) = (argument(args, 0), argument(args, 1)) {
        if divisor == 0 {
            return Err(arg_error(vm, 2, "fmod", "zero"));
        }
        let value = if divisor == -1 { 0 } else { dividend % divisor };
        return Ok(values![Value::Integer(value)]);
    }
    let first = check_number(vm, args, 0, "fmod")?.to_float();
    let second = check_number(vm, args, 1, "fmod")?.to_float();
    Ok(values![Value::Float(first % second)])
}

fn modf(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    if let Value::Integer(integer) = argument(args, 0) {
        return Ok(values![Value::Integer(integer), Value::Float(0.0)]);
    }
    let float = check_number(vm, args, 0, "modf")?.to_float();
    let integral = if float < 0.0 { float.ceil() } else { float.floor() };
    let fraction = if float == integral { 0.0 } else { float - integral };
    Ok(values![push_number_int(integral), Value::Float(fraction)])
}

fn float_argument(vm: &mut Vm, args: &[Value], name: &str) -> Result<f64, LuaError> {
    Ok(check_number(vm, args, 0, name)?.to_float())
}

fn sqrt(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(float_argument(vm, args, "sqrt")?.sqrt())])
}

fn frexp(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (mantissa, exponent) = split_float(float_argument(vm, args, "frexp")?);
    Ok(values![Value::Float(mantissa), Value::Integer(i64::from(exponent))])
}

fn ldexp(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let value = float_argument(vm, args, "ldexp")?;
    let exponent = check_integer(vm, args, 1, "ldexp")?;
    let exponent = exponent.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    Ok(values![Value::Float(scale_by_power_of_two(value, exponent))])
}

fn split_float(value: f64) -> (f64, i32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0);
    }
    let (value, adjust) = if value.abs() < f64::MIN_POSITIVE {
        (value * 2f64.powi(54), -54)
    } else {
        (value, 0)
    };
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mantissa = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1022_u64 << 52));
    (mantissa, exponent + adjust)
}

/// Scales in steps so that no intermediate power of two overflows.
fn scale_by_power_of_two(mut value: f64, mut exponent: i32) -> f64 {
    const STEP: i32 = 1000;
    while exponent > STEP {
        value *= 2f64.powi(STEP);
        exponent -= STEP;
    }
    while exponent < -STEP {
        value *= 2f64.powi(-STEP);
        exponent += STEP;
    }
    value * 2f64.powi(exponent)
}

fn exp(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::exp(float_argument(vm, args, "exp")?))])
}

fn sin(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::sin(float_argument(vm, args, "sin")?))])
}

fn cos(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::cos(float_argument(vm, args, "cos")?))])
}

fn tan(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::tan(float_argument(vm, args, "tan")?))])
}

fn asin(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::asin(float_argument(vm, args, "asin")?))])
}

fn acos(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(crate::math::acos(float_argument(vm, args, "acos")?))])
}

fn atan(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let y = float_argument(vm, args, "atan")?;
    let x = if argument(args, 1).is_nil() {
        1.0
    } else {
        check_number(vm, args, 1, "atan")?.to_float()
    };
    Ok(values![Value::Float(crate::math::atan2(y, x))])
}

fn log(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let x = float_argument(vm, args, "log")?;
    let result = if argument(args, 1).is_nil() {
        crate::math::ln(x)
    } else {
        let base = check_number(vm, args, 1, "log")?.to_float();
        if base == 2.0 {
            crate::math::log2(x)
        } else if base == 10.0 {
            crate::math::log10(x)
        } else {
            crate::math::ln(x) / crate::math::ln(base)
        }
    };
    Ok(values![Value::Float(result)])
}

fn deg(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(float_argument(vm, args, "deg")? * (180.0 / std::f64::consts::PI))])
}

fn rad(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(float_argument(vm, args, "rad")? * (std::f64::consts::PI / 180.0))])
}

impl Extremum {
    pub(crate) fn values(&self, out: &mut Values) {
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Extremum {
    arguments: Values,
    best: usize,
    next: usize,
    maximum: bool,
    started: bool,
}

fn maximum(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    extremum(vm, args, true)
}

fn minimum(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    extremum(vm, args, false)
}

fn extremum(vm: &mut Vm, args: &[Value], maximum: bool) -> Result<Step, LuaError> {
    if args.is_empty() {
        return Err(arg_error(vm, 1, if maximum { "max" } else { "min" }, "value expected"));
    }
    let machine = Machine::Extremum(Extremum {
        arguments: Values::from_slice(args),
        best: 0,
        next: 1,
        maximum,
        started: false,
    });
    run(vm, machine, Input::Start)
}

impl Extremum {
    pub fn advance(&mut self, input: Input) -> Next {
        if self.started && input.truthy() {
            self.best = self.next - 1;
        }
        self.started = true;
        if self.next >= self.arguments.len() {
            return Next::Done(values![self.arguments[self.best]]);
        }
        let candidate = self.arguments[self.next];
        let best = self.arguments[self.best];
        self.next += 1;
        let (left, right) = if self.maximum { (best, candidate) } else { (candidate, best) };
        Next::Perform(Op::Less { left, right })
    }
}

fn set_seed(vm: &mut Vm, first: u64, second: u64) {
    vm.random = [first, 0xff, second, 0];
    for _ in 0..16 {
        next_random(&mut vm.random);
    }
}

fn rotate_left(value: u64, count: u32) -> u64 {
    value.rotate_left(count)
}

fn next_random(state: &mut [u64; 4]) -> u64 {
    let state0 = state[0];
    let state1 = state[1];
    let state2 = state[2] ^ state0;
    let state3 = state[3] ^ state1;
    let result = rotate_left(state1.wrapping_mul(5), 7).wrapping_mul(9);
    state[0] = state0 ^ state3;
    state[1] = state1 ^ state2;
    state[2] = state2 ^ (state1 << 17);
    state[3] = rotate_left(state3, 45);
    result
}

fn project(mut random: u64, n: u64, state: &mut [u64; 4]) -> u64 {
    if n & n.wrapping_add(1) == 0 {
        return random & n;
    }
    let mut limit = n;
    for shift in [1, 2, 4, 8, 16, 32] {
        limit |= limit >> shift;
    }
    loop {
        random &= limit;
        if random <= n {
            return random;
        }
        random = next_random(state);
    }
}

fn random(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let value = next_random(&mut vm.random);
    let (low, up) = match args.len() {
        0 => return Ok(values![Value::Float((value >> 11) as f64 * 0.5f64.powi(53))]),
        1 => {
            let up = check_integer(vm, args, 0, "random")?;
            if up == 0 {
                return Ok(values![Value::Integer(value as i64)]);
            }
            (1, up)
        }
        2 => (check_integer(vm, args, 0, "random")?, check_integer(vm, args, 1, "random")?),
        _ => return Err(vm.runtime_error_text("wrong number of arguments")),
    };
    if low > up {
        return Err(arg_error(vm, 1, "random", "interval is empty"));
    }
    let offset = project(value, (up as u64).wrapping_sub(low as u64), &mut vm.random);
    Ok(values![Value::Integer(offset.wrapping_add(low as u64) as i64)])
}

fn randomseed(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let (first, second) = if args.is_empty() {
        (vm.host.system.now(), 0)
    } else {
        let second = if argument(args, 1).is_nil() {
            0
        } else {
            check_integer(vm, args, 1, "randomseed")?
        };
        (check_integer(vm, args, 0, "randomseed")?, second)
    };
    set_seed(vm, first as u64, second as u64);
    Ok(values![Value::Integer(first), Value::Integer(second)])
}
