
use super::{
    math::Extremum,
    package::RequireState,
    tablelib::{Concat, Insert, Move, Remove, Sort, Unpack},
};
use crate::{
    LuaError, values, Values, Vm,
    object::{Resume, Step},
    value::Value,
    vm::{IndexStep, SetStep, interrupt::Interrupts},
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Input {
    Start,
    Value(Value),
    Results(Values),
}

impl Input {
    pub fn value(&self) -> Value {
        match self {
            Input::Value(value) => *value,
            Input::Results(results) => results.first().copied().unwrap_or(Value::Nil),
            Input::Start => Value::Nil,
        }
    }

    pub fn truthy(&self) -> bool {
        !self.value().is_falsy()
    }

    pub fn results(&self) -> Values {
        match self {
            Input::Results(results) => results.clone(),
            Input::Value(value) => values![*value],
            Input::Start => Values::new(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Op {
    Get { object: Value, key: Value },
    Set { object: Value, key: Value, value: Value },
    Len(Value),
    Less { left: Value, right: Value },
    Equal { left: Value, right: Value },
    Call { function: Value, arguments: Values },
}

pub enum Next {
    Perform(Op),
    Done(Values),
}

pub enum Performed {
    Value(Value),
    Call { function: Value, arguments: Values },
}

impl Machine {
    pub(crate) fn values(&self, out: &mut Values) {
        match self {
            Machine::Insert(state) => state.values(out),
            Machine::Remove(state) => state.values(out),
            Machine::Concat(state) => state.values(out),
            Machine::Unpack(state) => state.values(out),
            Machine::Move(state) => state.values(out),
            Machine::Sort(state) => state.values(out),
            Machine::Extremum(state) => state.values(out),
            Machine::Require(state) => state.values(out),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Machine {
    Insert(Insert),
    Remove(Remove),
    Concat(Concat),
    Unpack(Unpack),
    Move(Move),
    Sort(Sort),
    Extremum(Extremum),
    Require(RequireState),
}

impl Machine {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self {
            Machine::Insert(machine) => machine.advance(vm, input),
            Machine::Remove(machine) => machine.advance(vm, input),
            Machine::Concat(machine) => machine.advance(vm, input),
            Machine::Unpack(machine) => machine.advance(vm, input),
            Machine::Move(machine) => Ok(machine.advance(input)),
            Machine::Sort(machine) => machine.advance(vm, input),
            Machine::Extremum(machine) => Ok(machine.advance(input)),
            Machine::Require(machine) => machine.advance(vm, input),
        }
    }
}

pub fn run(vm: &mut Vm, mut machine: Machine, mut input: Input) -> Result<Step, LuaError> {
    let mut interrupts = Interrupts::new(vm);
    loop {
        vm.check_interrupt(&mut interrupts)?;
        match machine.advance(vm, input)? {
            Next::Done(results) => return Ok(Step::Done(results)),
            Next::Perform(op) => match perform(vm, op)? {
                Performed::Value(value) => input = Input::Value(value),
                Performed::Call { function, arguments } => {
                    return Ok(Step::Call {
                        function,
                        arguments,
                        resume: Resume::Machine(Box::new(machine)),
                    });
                }
            },
        }
    }
}

pub fn resume(vm: &mut Vm, machine: Machine, values: Values) -> Result<Step, LuaError> {
    run(vm, machine, Input::Results(values))
}

fn perform(vm: &mut Vm, op: Op) -> Result<Performed, LuaError> {
    let outcome = match op {
        Op::Get { object, key } => chain(vm.index_chain(object, key)?),
        Op::Set { object, key, value } => match vm.store_step(object, key, value)? {
            SetStep::Done => Performed::Value(Value::Nil),
            SetStep::Call { function, arguments } => Performed::Call { function, arguments: arguments.into() },
        },
        Op::Len(value) => chain(vm.length_step(value)?),
        Op::Less { left, right } => chain(vm.order_step(left, right)?),
        Op::Equal { left, right } => chain(vm.equal_step(left, right)?),
        Op::Call { function, arguments } => Performed::Call { function, arguments },
    };
    Ok(outcome)
}

fn chain(step: IndexStep) -> Performed {
    match step {
        IndexStep::Value(value) => Performed::Value(value),
        IndexStep::Call { function, arguments } => Performed::Call { function, arguments: arguments.into() },
    }
}
