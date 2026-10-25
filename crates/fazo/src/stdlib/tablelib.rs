
use super::{
    arg_error, argument, check_integer,
    machine::{Input, Machine, Next, Op, run},
};
use crate::{
    LuaError, Values, values, Vm,
    arena::Ref,
    number::{self, Number, Rounding},
    object::{NativeFn, Step, SteppedFn},
    table::Table,
    value::Value,
    vm::{Event, interrupt::Interrupts},
};

const READ: u8 = 1;
const WRITE: u8 = 2;
const LENGTH: u8 = 4;

const MAX_INTEGER: i64 = i64::MAX;

const SORT_LIMIT: i64 = i32::MAX as i64;

const MAX_UNPACK_RESULTS: i64 = 999_000;

const SORT_RANDOM_PIVOT: u32 = 0x2545_f491;

const RANDOM_PIVOT_LIMIT: i64 = 100;

/// Follows `MAXHSIZE` in PUC Lua 5.5's ltable.c.
const MAX_HASH_SIZE: i64 = 1 << 30;

pub(super) fn open(vm: &mut Vm) {
    let functions: &[(&'static str, NativeFn)] = &[("create", create), ("pack", pack)];
    let stepped: &[(&'static str, SteppedFn)] = &[
        ("concat", concat),
        ("insert", insert),
        ("remove", remove),
        ("unpack", unpack),
        ("move", move_elements),
        ("sort", sort),
    ];
    let table = vm.new_table();
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    for (name, function) in stepped {
        let value = vm.new_stepped(name, *function);
        vm.set_field(table, name, value);
    }
    vm.set_global("table", Value::Table(table));
}

fn type_of_argument(vm: &mut Vm, args: &[Value], index: usize) -> String {
    match args.get(index) {
        Some(value) => vm.object_type_name(*value),
        None => "no value".to_string(),
    }
}

fn check_access(vm: &mut Vm, args: &[Value], index: usize, access: u8, name: &str) -> Result<(), LuaError> {
    let value = argument(args, index);
    if matches!(value, Value::Table(_)) && index < args.len() {
        return Ok(());
    }
    if let Some(metatable) = vm.metatable_of(value) {
        let mut events = Vec::new();
        if access & READ != 0 {
            events.push("__index");
        }
        if access & WRITE != 0 {
            events.push("__newindex");
        }
        if access & LENGTH != 0 {
            events.push("__len");
        }
        if events.iter().all(|event| !vm.raw_field(metatable, event).is_nil()) {
            return Ok(());
        }
    }
    let got = type_of_argument(vm, args, index);
    Err(arg_error(vm, index + 1, name, &format!("table expected, got {got}")))
}

fn integer_length(vm: &mut Vm, value: Value) -> Result<i64, LuaError> {
    let number = match value {
        Value::Integer(integer) => Some(Number::Integer(integer)),
        Value::Float(float) => Some(Number::Float(float)),
        Value::String(string) => number::parse_number(vm.heap_bytes(string)),
        _ => None,
    };
    match number.and_then(|number| number::to_integer(number, Rounding::Exact)) {
        Some(length) => Ok(length),
        None => Err(vm.runtime_error_text("object length is not an integer")),
    }
}

fn optional_integer(vm: &mut Vm, args: &[Value], index: usize, name: &str, default: i64) -> Result<i64, LuaError> {
    if argument(args, index).is_nil() {
        return Ok(default);
    }
    check_integer(vm, args, index, name)
}

fn optional_text(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Vec<u8>, LuaError> {
    match argument(args, index) {
        Value::Nil => Ok(Vec::new()),
        value @ (Value::String(_) | Value::Integer(_) | Value::Float(_)) => Ok(vm.to_display_bytes(value)),
        _ => {
            let got = type_of_argument(vm, args, index);
            Err(arg_error(vm, index + 1, name, &format!("string expected, got {got}")))
        }
    }
}

fn start(vm: &mut Vm, machine: Machine) -> Result<Step, LuaError> {
    run(vm, machine, Input::Start)
}

fn get(table: Value, index: i64) -> Op {
    Op::Get {
        object: table,
        key: Value::Integer(index),
    }
}

fn set(table: Value, index: i64, value: Value) -> Op {
    Op::Set {
        object: table,
        key: Value::Integer(index),
        value,
    }
}

fn create(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let sequence = check_integer(vm, args, 0, "create")?;
    let rest = optional_integer(vm, args, 1, "create", 0)?;
    let limit = i64::from(i32::MAX);
    if !(0..=limit).contains(&sequence) {
        return Err(arg_error(vm, 1, "create", "out of range"));
    }
    if !(0..=limit).contains(&rest) {
        return Err(arg_error(vm, 2, "create", "out of range"));
    }
    if rest > MAX_HASH_SIZE {
        return Err(vm.runtime_error_text("table overflow"));
    }
    vm.reserve(crate::table::Table::sized_bytes(sequence as usize, rest as usize))?;
    let table = vm.new_table_with_sizes(sequence as usize, rest as usize)?;
    Ok(values![Value::Table(table)])
}

fn pack(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let table = vm.new_table();
    for (offset, value) in args.iter().enumerate().rev() {
        vm.table_set(table, Value::Integer(offset as i64 + 1), *value)?;
    }
    vm.set_field(table, "n", Value::Integer(args.len() as i64));
    Ok(values![Value::Table(table)])
}

impl Insert {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.table);
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Insert {
    table: Value,
    arguments: Values,
    position: i64,
    next: i64,
    phase: InsertPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum InsertPhase {
    Start,
    Measured,
    Shift,
    Moved,
    Stored,
}

fn insert(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_access(vm, args, 0, READ | WRITE | LENGTH, "insert")?;
    let machine = Machine::Insert(Insert {
        table: args[0],
        arguments: Values::from_slice(args),
        position: 0,
        next: 0,
        phase: InsertPhase::Start,
    });
    start(vm, machine)
}

impl Insert {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self.phase {
            InsertPhase::Start => {
                self.phase = InsertPhase::Measured;
                Ok(Next::Perform(Op::Len(self.table)))
            }
            InsertPhase::Measured => self.place(vm, input.value()),
            InsertPhase::Shift => Ok(self.shift()),
            InsertPhase::Moved => Ok(self.move_one(input.value())),
            InsertPhase::Stored => Ok(Next::Done(Values::new())),
        }
    }

    fn place(&mut self, vm: &mut Vm, length: Value) -> Result<Next, LuaError> {
        let end = integer_length(vm, length)?.wrapping_add(1);
        self.position = match self.arguments.len() {
            2 => end,
            3 => {
                let position = check_integer(vm, &self.arguments, 1, "insert")?;
                if (position as u64).wrapping_sub(1) >= end as u64 {
                    return Err(arg_error(vm, 2, "insert", "position out of bounds"));
                }
                position
            }
            _ => return Err(vm.runtime_error_text("wrong number of arguments to 'insert'")),
        };
        self.next = end;
        self.phase = InsertPhase::Shift;
        Ok(self.shift())
    }

    fn shift(&mut self) -> Next {
        if self.next > self.position {
            self.phase = InsertPhase::Moved;
            return Next::Perform(get(self.table, self.next - 1));
        }
        self.phase = InsertPhase::Stored;
        let value = self.arguments[self.arguments.len() - 1];
        Next::Perform(set(self.table, self.position, value))
    }

    fn move_one(&mut self, value: Value) -> Next {
        self.phase = InsertPhase::Shift;
        let op = set(self.table, self.next, value);
        self.next -= 1;
        Next::Perform(op)
    }
}

impl Remove {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.table);
        out.push(self.result);
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Remove {
    table: Value,
    arguments: Values,
    size: i64,
    position: i64,
    result: Value,
    phase: RemovePhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum RemovePhase {
    Start,
    Measured,
    Fetched,
    Loop,
    Copied,
    Cleared,
}

fn remove(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_access(vm, args, 0, READ | WRITE | LENGTH, "remove")?;
    let machine = Machine::Remove(Remove {
        table: args[0],
        arguments: Values::from_slice(args),
        size: 0,
        position: 0,
        result: Value::Nil,
        phase: RemovePhase::Start,
    });
    start(vm, machine)
}

impl Remove {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self.phase {
            RemovePhase::Start => {
                self.phase = RemovePhase::Measured;
                Ok(Next::Perform(Op::Len(self.table)))
            }
            RemovePhase::Measured => self.begin(vm, input.value()),
            RemovePhase::Fetched => {
                self.result = input.value();
                self.phase = RemovePhase::Loop;
                Ok(self.step_loop())
            }
            RemovePhase::Loop => Ok(self.step_loop()),
            RemovePhase::Copied => {
                self.phase = RemovePhase::Loop;
                let op = set(self.table, self.position, input.value());
                self.position += 1;
                Ok(Next::Perform(op))
            }
            RemovePhase::Cleared => Ok(Next::Done(values![self.result])),
        }
    }

    fn begin(&mut self, vm: &mut Vm, length: Value) -> Result<Next, LuaError> {
        self.size = integer_length(vm, length)?;
        self.position = optional_integer(vm, &self.arguments, 1, "remove", self.size)?;
        if self.position != self.size && (self.position as u64).wrapping_sub(1) > self.size as u64 {
            return Err(arg_error(vm, 2, "remove", "position out of bounds"));
        }
        self.phase = RemovePhase::Fetched;
        Ok(Next::Perform(get(self.table, self.position)))
    }

    fn step_loop(&mut self) -> Next {
        if self.position < self.size {
            self.phase = RemovePhase::Copied;
            return Next::Perform(get(self.table, self.position + 1));
        }
        self.phase = RemovePhase::Cleared;
        Next::Perform(set(self.table, self.position, Value::Nil))
    }
}

impl Concat {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.table);
        out.extend_from_slice(&self.arguments);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Concat {
    table: Value,
    arguments: Values,
    #[serde(with = "crate::bytes")]
    separator: Vec<u8>,
    index: i64,
    last: i64,
    #[serde(with = "crate::bytes")]
    buffer: Vec<u8>,
    phase: ConcatPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum ConcatPhase {
    Start,
    Measured,
    Element,
    Piece,
}

fn concat(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_access(vm, args, 0, READ | LENGTH, "concat")?;
    let machine = Machine::Concat(Concat {
        table: args[0],
        arguments: Values::from_slice(args),
        separator: Vec::new(),
        index: 1,
        last: 0,
        buffer: Vec::new(),
        phase: ConcatPhase::Start,
    });
    start(vm, machine)
}

impl Concat {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self.phase {
            ConcatPhase::Start => {
                self.phase = ConcatPhase::Measured;
                Ok(Next::Perform(Op::Len(self.table)))
            }
            ConcatPhase::Measured => self.begin(vm, input.value()),
            ConcatPhase::Element => Ok(self.element(vm)),
            ConcatPhase::Piece => self.piece(vm, input.value()),
        }
    }

    fn begin(&mut self, vm: &mut Vm, length: Value) -> Result<Next, LuaError> {
        let last = integer_length(vm, length)?;
        self.separator = optional_text(vm, &self.arguments, 1, "concat")?;
        self.index = optional_integer(vm, &self.arguments, 2, "concat", 1)?;
        self.last = optional_integer(vm, &self.arguments, 3, "concat", last)?;
        self.phase = ConcatPhase::Element;
        Ok(self.element(vm))
    }

    fn element(&mut self, vm: &mut Vm) -> Next {
        if self.index > self.last {
            return Next::Done(values![vm.string_value(&self.buffer)]);
        }
        self.phase = ConcatPhase::Piece;
        Next::Perform(get(self.table, self.index))
    }

    fn piece(&mut self, vm: &mut Vm, value: Value) -> Result<Next, LuaError> {
        match value {
            Value::String(string) => self.buffer.extend_from_slice(vm.heap_bytes(string)),
            Value::Integer(_) | Value::Float(_) => {
                let text = vm.to_display_bytes(value);
                self.buffer.extend_from_slice(&text);
            }
            other => {
                let message = format!(
                    "invalid value ({}) at index {} in table for 'concat'",
                    other.type_name(),
                    self.index
                );
                return Err(vm.runtime_error_text(&message));
            }
        }
        if self.index == self.last {
            return Ok(Next::Done(values![vm.string_value(&self.buffer)]));
        }
        self.buffer.extend_from_slice(&self.separator);
        self.index += 1;
        Ok(self.element(vm))
    }
}

impl Unpack {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.table);
        out.extend_from_slice(&self.results);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Unpack {
    table: Value,
    next: i64,
    given_last: Option<i64>,
    last: i64,
    results: Values,
    phase: UnpackPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum UnpackPhase {
    Start,
    Measured,
    Stored,
}

fn unpack(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let first = optional_integer(vm, args, 1, "unpack", 1)?;
    let last = if argument(args, 2).is_nil() {
        None
    } else {
        Some(check_integer(vm, args, 2, "unpack")?)
    };
    let machine = Machine::Unpack(Unpack {
        table: argument(args, 0),
        next: first,
        given_last: last,
        last: 0,
        results: Values::new(),
        phase: UnpackPhase::Start,
    });
    start(vm, machine)
}

impl Unpack {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        match self.phase {
            UnpackPhase::Start => match self.given_last {
                Some(last) => self.begin(vm, last),
                None => {
                    self.phase = UnpackPhase::Measured;
                    Ok(Next::Perform(Op::Len(self.table)))
                }
            },
            UnpackPhase::Measured => {
                let last = integer_length(vm, input.value())?;
                self.begin(vm, last)
            }
            UnpackPhase::Stored => {
                self.results.push(input.value());
                if self.next == self.last {
                    return Ok(Next::Done(std::mem::take(&mut self.results)));
                }
                self.next += 1;
                Ok(self.fetch())
            }
        }
    }

    fn begin(&mut self, vm: &mut Vm, last: i64) -> Result<Next, LuaError> {
        self.last = last;
        if self.next > last {
            return Ok(Next::Done(Values::new()));
        }
        let count = (last as u64).wrapping_sub(self.next as u64);
        if count >= i32::MAX as u64 || count as i64 + 1 > MAX_UNPACK_RESULTS {
            return Err(vm.runtime_error_text("too many results to unpack"));
        }
        Ok(self.fetch())
    }

    fn fetch(&mut self) -> Next {
        self.phase = UnpackPhase::Stored;
        Next::Perform(get(self.table, self.next))
    }
}

impl Move {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.source);
        out.push(self.destination);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Move {
    source: Value,
    destination: Value,
    first: i64,
    last: i64,
    count: i64,
    target: i64,
    forward: bool,
    destination_is_first: bool,
    cursor: i64,
    phase: MovePhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum MovePhase {
    Decide,
    Compared,
    Loop,
    Put,
}

fn move_elements(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    let first = check_integer(vm, args, 1, "move")?;
    let last = check_integer(vm, args, 2, "move")?;
    let target = check_integer(vm, args, 3, "move")?;
    let destination_index = if argument(args, 4).is_nil() { 0 } else { 4 };
    let destination = argument(args, destination_index);
    check_access(vm, args, 0, READ, "move")?;
    check_access(vm, args, destination_index, WRITE, "move")?;
    if last < first {
        return Ok(Step::Done(values![destination]));
    }
    if first <= 0 && last >= MAX_INTEGER + first {
        return Err(arg_error(vm, 3, "move", "too many elements to move"));
    }
    let count = last - first + 1;
    if target > MAX_INTEGER - count + 1 {
        return Err(arg_error(vm, 4, "move", "destination wrap around"));
    }
    let machine = Machine::Move(Move {
        source: args[0],
        destination,
        first,
        last,
        count,
        target,
        forward: true,
        destination_is_first: destination_index == 0,
        cursor: 0,
        phase: MovePhase::Decide,
    });
    start(vm, machine)
}

impl Move {
    pub fn advance(&mut self, input: Input) -> Next {
        match self.phase {
            MovePhase::Decide => self.decide(),
            MovePhase::Compared => {
                self.forward = !input.truthy();
                self.start_loop();
                self.step_loop()
            }
            MovePhase::Loop => self.step_loop(),
            MovePhase::Put => {
                let op = set(self.destination_table(), self.target + self.cursor, input.value());
                self.cursor += if self.forward { 1 } else { -1 };
                self.phase = MovePhase::Loop;
                Next::Perform(op)
            }
        }
    }

    fn destination_table(&self) -> Value {
        self.destination
    }

    fn decide(&mut self) -> Next {
        if self.target > self.last || self.target <= self.first {
            self.start_loop();
            return self.step_loop();
        }
        if self.destination_is_first {
            self.forward = false;
            self.start_loop();
            return self.step_loop();
        }
        self.phase = MovePhase::Compared;
        Next::Perform(Op::Equal {
            left: self.source,
            right: self.destination,
        })
    }

    fn start_loop(&mut self) {
        self.phase = MovePhase::Loop;
        self.cursor = if self.forward { 0 } else { self.count - 1 };
    }

    fn step_loop(&mut self) -> Next {
        if self.cursor < 0 || self.cursor >= self.count {
            return Next::Done(values![self.destination]);
        }
        self.phase = MovePhase::Put;
        Next::Perform(get(self.source, self.first + self.cursor))
    }
}

impl Sort {
    pub(crate) fn values(&self, out: &mut Values) {
        out.push(self.table);
        out.push(self.comparator);
        for frame in &self.frames {
            frame.values(out);
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Sort {
    table: Value,
    comparator: Value,
    phase: SortPhase,
    frames: Vec<Frame>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum SortPhase {
    Start,
    Length,
    Running,
}

impl Frame {
    fn values(&self, out: &mut Values) {
        out.extend([self.low, self.high, self.middle, self.pivot, self.held_i, self.held_j]);
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Frame {
    lo: i64,
    up: i64,
    rnd: u32,
    stage: Stage,
    p: i64,
    i: i64,
    j: i64,
    n: i64,
    low: Value,
    high: Value,
    middle: Value,
    pivot: Value,
    held_i: Value,
    held_j: Value,
}

impl Frame {
    fn new(lo: i64, up: i64, rnd: u32) -> Self {
        Self {
            lo,
            up,
            rnd,
            stage: Stage::Enter,
            p: 0,
            i: 0,
            j: 0,
            n: 0,
            low: Value::Nil,
            high: Value::Nil,
            middle: Value::Nil,
            pivot: Value::Nil,
            held_i: Value::Nil,
            held_j: Value::Nil,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum Stage {
    Enter,
    AwaitLow,
    AwaitHigh,
    CompareEnds,
    SwapEndsHigh,
    AfterEnds,
    AwaitMiddle,
    AwaitLowForPivot,
    ComparePivotLow,
    SwapPivotLow,
    AwaitHighForPivot,
    ComparePivotHigh,
    SwapPivotHigh,
    AfterPivot,
    AwaitPivot,
    AwaitPenultimate,
    SwapPenultimate,
    PartitionStart,
    PartitionStep,
    PartitionRead,
    PartitionCompareI,
    PartitionJStep,
    PartitionReadJ,
    PartitionCompareJ,
    PartitionDecide,
    PartitionSwapTail,
    PartitionPlaceCenter,
    PartitionDone,
    AfterChild,
}

enum Flow {
    Perform(Op),
    Pass,
    Split(Frame),
    Finish,
}

fn sort(vm: &mut Vm, args: &[Value]) -> Result<Step, LuaError> {
    check_access(vm, args, 0, READ | WRITE | LENGTH, "sort")?;
    let machine = Machine::Sort(Sort {
        table: args[0],
        comparator: argument(args, 1),
        phase: SortPhase::Start,
        frames: Vec::new(),
    });
    start(vm, machine)
}

impl Sort {
    pub fn advance(&mut self, vm: &mut Vm, input: Input) -> Result<Next, LuaError> {
        let mut input = input.value();
        match self.phase {
            SortPhase::Start => {
                self.phase = SortPhase::Length;
                return Ok(Next::Perform(Op::Len(self.table)));
            }
            SortPhase::Length => {
                let count = integer_length(vm, input)?;
                if count <= 1 {
                    return Ok(Next::Done(Values::new()));
                }
                if count >= SORT_LIMIT {
                    return Err(arg_error(vm, 1, "sort", "array too big"));
                }
                if !self.comparator.is_nil() && !self.comparator.is_function() {
                    let got = vm.object_type_name(self.comparator);
                    return Err(arg_error(vm, 2, "sort", &format!("function expected, got {got}")));
                }
                if self.comparator.is_nil() && sort_in_place(vm, self.table, count)? {
                    return Ok(Next::Done(Values::new()));
                }
                self.frames.push(Frame::new(1, count, 0));
                self.phase = SortPhase::Running;
                input = Value::Nil;
            }
            SortPhase::Running => {}
        }
        self.run(vm, input)
    }

    fn run(&mut self, vm: &mut Vm, mut input: Value) -> Result<Next, LuaError> {
        let mut interrupts = Interrupts::new(vm);
        loop {
            let Some(mut frame) = self.frames.pop() else {
                return Ok(Next::Done(Values::new()));
            };
            let flow = self.transition(vm, &mut frame, input)?;
            input = Value::Nil;
            match flow {
                Flow::Perform(op) => {
                    self.frames.push(frame);
                    let Some(value) = direct(vm, &op)? else {
                        return Ok(Next::Perform(op));
                    };
                    vm.check_interrupt(&mut interrupts)?;
                    input = value;
                }
                Flow::Pass => self.frames.push(frame),
                Flow::Split(child) => {
                    self.frames.push(frame);
                    self.frames.push(child);
                }
                Flow::Finish => {}
            }
        }
    }

    fn compare(&self, left: Value, right: Value) -> Op {
        if self.comparator.is_nil() {
            return Op::Less { left, right };
        }
        Op::Call {
            function: self.comparator,
            arguments: values![left, right],
        }
    }

    fn get(&self, index: i64) -> Flow {
        Flow::Perform(get(self.table, index))
    }

    fn set(&self, index: i64, value: Value) -> Flow {
        Flow::Perform(set(self.table, index, value))
    }

    fn transition(&self, vm: &mut Vm, frame: &mut Frame, input: Value) -> Result<Flow, LuaError> {
        let flow = match frame.stage {
            Stage::Enter => {
                if frame.lo >= frame.up {
                    return Ok(Flow::Finish);
                }
                frame.stage = Stage::AwaitLow;
                self.get(frame.lo)
            }
            Stage::AwaitLow => {
                frame.low = input;
                frame.stage = Stage::AwaitHigh;
                self.get(frame.up)
            }
            Stage::AwaitHigh => {
                frame.high = input;
                frame.stage = Stage::CompareEnds;
                Flow::Perform(self.compare(frame.high, frame.low))
            }
            Stage::CompareEnds => {
                if input.is_falsy() {
                    frame.stage = Stage::AfterEnds;
                    return Ok(Flow::Pass);
                }
                frame.stage = Stage::SwapEndsHigh;
                self.set(frame.lo, frame.high)
            }
            Stage::SwapEndsHigh => {
                frame.stage = Stage::AfterEnds;
                self.set(frame.up, frame.low)
            }
            Stage::AfterEnds => {
                if frame.up - frame.lo == 1 {
                    return Ok(Flow::Finish);
                }
                frame.p = pivot_index(frame.lo, frame.up, frame.rnd);
                frame.stage = Stage::AwaitMiddle;
                self.get(frame.p)
            }
            Stage::AwaitMiddle => {
                frame.middle = input;
                frame.stage = Stage::AwaitLowForPivot;
                self.get(frame.lo)
            }
            Stage::AwaitLowForPivot => {
                frame.low = input;
                frame.stage = Stage::ComparePivotLow;
                Flow::Perform(self.compare(frame.middle, frame.low))
            }
            Stage::ComparePivotLow => {
                if input.is_falsy() {
                    frame.stage = Stage::AwaitHighForPivot;
                    self.get(frame.up)
                } else {
                    frame.stage = Stage::SwapPivotLow;
                    self.set(frame.p, frame.low)
                }
            }
            Stage::SwapPivotLow => {
                frame.stage = Stage::AfterPivot;
                self.set(frame.lo, frame.middle)
            }
            Stage::AwaitHighForPivot => {
                frame.high = input;
                frame.stage = Stage::ComparePivotHigh;
                Flow::Perform(self.compare(frame.high, frame.middle))
            }
            Stage::ComparePivotHigh => {
                if input.is_falsy() {
                    frame.stage = Stage::AfterPivot;
                    return Ok(Flow::Pass);
                }
                frame.stage = Stage::SwapPivotHigh;
                self.set(frame.p, frame.high)
            }
            Stage::SwapPivotHigh => {
                frame.stage = Stage::AfterPivot;
                self.set(frame.up, frame.middle)
            }
            Stage::AfterPivot => {
                if frame.up - frame.lo == 2 {
                    return Ok(Flow::Finish);
                }
                frame.stage = Stage::AwaitPivot;
                self.get(frame.p)
            }
            Stage::AwaitPivot => {
                frame.pivot = input;
                frame.stage = Stage::AwaitPenultimate;
                self.get(frame.up - 1)
            }
            Stage::AwaitPenultimate => {
                frame.stage = Stage::SwapPenultimate;
                self.set(frame.p, input)
            }
            Stage::SwapPenultimate => {
                frame.stage = Stage::PartitionStart;
                self.set(frame.up - 1, frame.pivot)
            }
            Stage::PartitionStart => {
                frame.i = frame.lo;
                frame.j = frame.up - 1;
                frame.stage = Stage::PartitionStep;
                Flow::Pass
            }
            Stage::PartitionStep => {
                frame.i += 1;
                frame.stage = Stage::PartitionRead;
                self.get(frame.i)
            }
            Stage::PartitionRead => {
                frame.held_i = input;
                frame.stage = Stage::PartitionCompareI;
                Flow::Perform(self.compare(frame.held_i, frame.pivot))
            }
            Stage::PartitionCompareI => {
                if input.is_falsy() {
                    frame.stage = Stage::PartitionJStep;
                    return Ok(Flow::Pass);
                }
                if frame.i == frame.up - 1 {
                    return Err(vm.runtime_error_text("invalid order function for sorting"));
                }
                frame.stage = Stage::PartitionStep;
                Flow::Pass
            }
            Stage::PartitionJStep => {
                frame.j -= 1;
                frame.stage = Stage::PartitionReadJ;
                self.get(frame.j)
            }
            Stage::PartitionReadJ => {
                frame.held_j = input;
                frame.stage = Stage::PartitionCompareJ;
                Flow::Perform(self.compare(frame.pivot, frame.held_j))
            }
            Stage::PartitionCompareJ => {
                if input.is_falsy() {
                    frame.stage = Stage::PartitionDecide;
                    return Ok(Flow::Pass);
                }
                if frame.j < frame.i {
                    return Err(vm.runtime_error_text("invalid order function for sorting"));
                }
                frame.stage = Stage::PartitionJStep;
                Flow::Pass
            }
            Stage::PartitionDecide => {
                if frame.j < frame.i {
                    frame.stage = Stage::PartitionPlaceCenter;
                    self.set(frame.up - 1, frame.held_i)
                } else {
                    frame.stage = Stage::PartitionSwapTail;
                    self.set(frame.i, frame.held_j)
                }
            }
            Stage::PartitionSwapTail => {
                frame.stage = Stage::PartitionStep;
                self.set(frame.j, frame.held_i)
            }
            Stage::PartitionPlaceCenter => {
                frame.stage = Stage::PartitionDone;
                self.set(frame.i, frame.pivot)
            }
            Stage::PartitionDone => split(frame),
            Stage::AfterChild => {
                if (frame.up - frame.lo) / 128 > frame.n {
                    frame.rnd = SORT_RANDOM_PIVOT;
                }
                frame.stage = Stage::Enter;
                Flow::Pass
            }
        };
        Ok(flow)
    }
}

fn direct(vm: &mut Vm, op: &Op) -> Result<Option<Value>, LuaError> {
    match *op {
        Op::Get { object: Value::Table(table), key } if vm.lacks_event(table, Event::Index) => {
            Ok(Some(vm.table_get(table, key)))
        }
        Op::Set { object: Value::Table(table), key, value } if vm.lacks_event(table, Event::NewIndex) => {
            vm.raw_set(table, key, value)?;
            Ok(Some(Value::Nil))
        }
        Op::Less { left, right } => Ok(primitive_less(vm, left, right).map(Value::Boolean)),
        _ => Ok(None),
    }
}

fn primitive_less(vm: &Vm, left: Value, right: Value) -> Option<bool> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a < b),
        (Value::Float(a), Value::Float(b)) => Some(a < b),
        (Value::Integer(a), Value::Float(b)) => Some(number::less_integer_float(a, b)),
        (Value::Float(a), Value::Integer(b)) => Some(number::less_float_integer(a, b)),
        (Value::String(a), Value::String(b)) => Some(vm.heap_bytes(a) < vm.heap_bytes(b)),
        _ => None,
    }
}

fn split(frame: &mut Frame) -> Flow {
    let p = frame.i;
    let child = if p - frame.lo < frame.up - p {
        let child = Frame::new(frame.lo, p - 1, frame.rnd);
        frame.n = p - frame.lo;
        frame.lo = p + 1;
        child
    } else {
        let child = Frame::new(p + 1, frame.up, frame.rnd);
        frame.n = frame.up - p;
        frame.up = p - 1;
        child
    };
    frame.stage = Stage::AfterChild;
    Flow::Split(child)
}

fn pivot_index(lo: i64, up: i64, rnd: u32) -> i64 {
    if up - lo < RANDOM_PIVOT_LIMIT || rnd == 0 {
        return (lo + up) / 2;
    }
    let quarter = ((up - lo) / 4) as u64;
    let offset = u64::from(rnd) % (quarter * 2);
    lo + offset as i64 + quarter as i64
}

fn sort_in_place(vm: &mut Vm, object: Value, count: i64) -> Result<bool, LuaError> {
    let Value::Table(table) = object else {
        return Ok(false);
    };
    if !vm.lacks_event(table, Event::Index) || !vm.lacks_event(table, Event::NewIndex) {
        return Ok(false);
    }
    if !homogeneous_primitives(vm, table, count) {
        return Ok(false);
    }
    let mut interrupts = Interrupts::new(vm);
    sort_range(vm, table, &mut interrupts, 1, count, 0)?;
    Ok(true)
}

#[derive(PartialEq, Eq)]
enum PrimitiveKind {
    Number,
    Text,
}

fn homogeneous_primitives(vm: &Vm, table: Ref<Table>, count: i64) -> bool {
    let mut kind = None;
    for index in 1..=count {
        let this = match vm.table_get(table, Value::Integer(index)) {
            Value::Integer(_) | Value::Float(_) => PrimitiveKind::Number,
            Value::String(_) => PrimitiveKind::Text,
            _ => return false,
        };
        match &kind {
            None => kind = Some(this),
            Some(first) if *first == this => {}
            Some(_) => return false,
        }
    }
    true
}

fn sort_range(
    vm: &mut Vm,
    table: Ref<Table>,
    interrupts: &mut Interrupts,
    mut lo: i64,
    mut up: i64,
    mut rnd: u32,
) -> Result<(), LuaError> {
    while lo < up {
        vm.check_interrupt(interrupts)?;
        order_ends(vm, table, lo, up)?;
        if up - lo == 1 {
            return Ok(());
        }
        let p = pivot_index(lo, up, rnd);
        order_pivot(vm, table, lo, p, up)?;
        if up - lo == 2 {
            return Ok(());
        }
        let pivot = read(vm, table, p);
        let penultimate = read(vm, table, up - 1);
        write(vm, table, p, penultimate)?;
        write(vm, table, up - 1, pivot)?;
        let i = partition(vm, table, lo, up, pivot)?;
        if i - lo < up - i {
            sort_range(vm, table, interrupts, lo, i - 1, rnd)?;
            let smaller = i - lo;
            lo = i + 1;
            rnd = rnd_after(lo, up, smaller, rnd);
        } else {
            sort_range(vm, table, interrupts, i + 1, up, rnd)?;
            let smaller = up - i;
            up = i - 1;
            rnd = rnd_after(lo, up, smaller, rnd);
        }
    }
    Ok(())
}

fn rnd_after(lo: i64, up: i64, smaller: i64, rnd: u32) -> u32 {
    if (up - lo) / 128 > smaller {
        return SORT_RANDOM_PIVOT;
    }
    rnd
}

fn order_ends(vm: &mut Vm, table: Ref<Table>, lo: i64, up: i64) -> Result<(), LuaError> {
    let low = read(vm, table, lo);
    let high = read(vm, table, up);
    if less(vm, high, low)? {
        write(vm, table, lo, high)?;
        write(vm, table, up, low)?;
    }
    Ok(())
}

fn order_pivot(vm: &mut Vm, table: Ref<Table>, lo: i64, p: i64, up: i64) -> Result<(), LuaError> {
    let middle = read(vm, table, p);
    let low = read(vm, table, lo);
    if less(vm, middle, low)? {
        write(vm, table, p, low)?;
        write(vm, table, lo, middle)?;
        return Ok(());
    }
    let high = read(vm, table, up);
    if less(vm, high, middle)? {
        write(vm, table, p, high)?;
        write(vm, table, up, middle)?;
    }
    Ok(())
}

fn partition(vm: &mut Vm, table: Ref<Table>, lo: i64, up: i64, pivot: Value) -> Result<i64, LuaError> {
    let mut i = lo;
    let mut j = up - 1;
    loop {
        let held_i = loop {
            i += 1;
            let held = read(vm, table, i);
            if !less(vm, held, pivot)? {
                break held;
            }
            if i == up - 1 {
                return Err(vm.runtime_error_text("invalid order function for sorting"));
            }
        };
        let held_j = loop {
            j -= 1;
            let held = read(vm, table, j);
            if !less(vm, pivot, held)? {
                break held;
            }
            if j < i {
                return Err(vm.runtime_error_text("invalid order function for sorting"));
            }
        };
        if j < i {
            write(vm, table, up - 1, held_i)?;
            write(vm, table, i, pivot)?;
            return Ok(i);
        }
        write(vm, table, i, held_j)?;
        write(vm, table, j, held_i)?;
    }
}

fn read(vm: &Vm, table: Ref<Table>, index: i64) -> Value {
    vm.table_get(table, Value::Integer(index))
}

fn write(vm: &mut Vm, table: Ref<Table>, index: i64, value: Value) -> Result<(), LuaError> {
    vm.raw_set(table, Value::Integer(index), value)
}

fn less(vm: &mut Vm, left: Value, right: Value) -> Result<bool, LuaError> {
    match primitive_less(vm, left, right) {
        Some(result) => Ok(result),
        None => Err(vm.order_error(left, right)),
    }
}
