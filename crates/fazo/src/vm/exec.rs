use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use super::{
    Builtin, CallRequest, CallStart, Event, MAX_CALL_METAMETHODS, RunScope, UpvaluesRef, fast, ic::Probe, interrupt::interrupt_requested,
    naming::Operand,
    ops::{Operands, arithmetic_target},
    quick::{self, int_loop_count},
    quicken::Dispatch,
    window::{Passed, Record, Window, opcode_of},
};
use crate::{
    LuaError, Vm,
    session::{LOOP_CHARGE, take_request},
    arena::Ref,
    instruction::{ArithOp, Instruction, UnaryOp},
    number::{self, Number, Rounding},
    object::{Captured, Closure, Frame, LuaFrame, Pending, status as frame_status},
    opcode::Opcode,
    proto::Proto,
    table::{SetError, Table},
    value::Value,
};

#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod direct;
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use direct::Direct;

enum StackWindow {
    Registers(usize),
    Multret,
}

/// Follows `MAXARG_vC + 1` in lopcodes.h.
const SETLIST_BLOCK: u32 = 1 << 10;

/// Keeps debug builds from giving each `?` a stack slot for its `Try` temporaries.
macro_rules! ok {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return Err(error),
        }
    };
}

macro_rules! fast_arith {
    ($window:ident, $ip:ident, $target:expr, $op:expr, $left:expr, $right:expr) => {{
        let (left, right) = ($left, $right);
        match (left, right) {
            (Value::Integer(x), Value::Integer(y)) => {
                if let Some(value) = fast::integer_result($op, x, y) {
                    $window.set_register($target, Value::Integer(value));
                    $ip = $ip.add(2);
                    continue;
                }
            }
            (Value::Float(x), Value::Float(y)) if !super::ops::is_bitwise($op) => {
                $window.set_register($target, Value::Float(number::float_arith($op, x, y)));
                $ip = $ip.add(2);
                continue;
            }
            _ => {}
        }
        if let Some(value) = fast::arith($op, left, right) {
            $window.set_register($target, value);
            $ip = $ip.add(1);
        }
        $ip = $ip.add(1);
        continue;
    }};
}

// Decoding fields one at a time keeps debug builds from giving every arm an Instruction temporary.
macro_rules! decode {
    ($word:ident, NewTable $fields:tt) => {
        let Instruction::NewTable $fields = crate::encoding::operands(Opcode::NewTable, $word) else {
            unreachable!()
        };
    };
    ($word:ident, SetList $fields:tt) => {
        let Instruction::SetList $fields = crate::encoding::operands(Opcode::SetList, $word) else {
            unreachable!()
        };
    };
    ($word:ident, $op:ident { $($field:ident),* $(, ..)? }) => {
        $(let $field = crate::encoding::field::$field($word);)*
    };
    ($word:ident, $op:ident { .. }) => {};
}

/// Follows `returning` in `luaV_execute` (lvm.c). No request point follows: a run of returns is bounded by the depth its calls reached.
macro_rules! return_inline {
    ($vm:ident, $window:ident, $ip:ident, $scope:ident, $charges:ident, $first:expr, $count:expr) => {{
        if let Some((next, resume)) = $vm.return_into_caller(&$window, $first, $count, $scope) {
            if $charges && $vm.charges() && next.charge_spent(crate::jit::ENTRY_CHARGE) && ok!($vm.jit_budget_spent(next.closure(), next.base(), next.proto(), resume, next.registers())) {
                break;
            }
            $window = next;
            $ip = $window.ip_at(resume);
            continue;
        }
    }};
}

/// Follows `MAXASIZE` in ltable.c.
const MAX_ARRAY_SIZE: u64 = 1 << 31;
const MAX_HASH_LOG: u8 = 31;

fn hash_size(log_plus_one: u8) -> usize {
    if log_plus_one == 0 { 0 } else { 1 << (log_plus_one - 1) }
}

fn arith_of_event(event: u8) -> Option<ArithOp> {
    Some(match event {
        6 => ArithOp::Add,
        7 => ArithOp::Sub,
        8 => ArithOp::Mul,
        9 => ArithOp::Mod,
        10 => ArithOp::Pow,
        11 => ArithOp::Div,
        12 => ArithOp::IDiv,
        13 => ArithOp::BAnd,
        14 => ArithOp::BOr,
        15 => ArithOp::BXor,
        16 => ArithOp::Shl,
        17 => ArithOp::Shr,
        _ => return None,
    })
}

fn gc_stack_window(instruction: Instruction) -> Option<StackWindow> {
    match instruction {
        Instruction::NewTable { a, .. } | Instruction::Closure { a, .. } => Some(StackWindow::Registers(usize::from(a))),
        Instruction::Concat { a, b } => Some(StackWindow::Registers(usize::from(a) + usize::from(b))),
        Instruction::Vararg { a, .. } => Some(StackWindow::Registers(usize::from(a))),
        Instruction::SetList { a, b, .. } => {
            if b == 0 {
                Some(StackWindow::Multret)
            } else {
                Some(StackWindow::Registers(usize::from(a) + usize::from(b) + 1))
            }
        }
        Instruction::Call { a, b, .. } | Instruction::TailCall { a, b, .. } => {
            if b == 0 {
                Some(StackWindow::Multret)
            } else {
                Some(StackWindow::Registers(usize::from(a) + usize::from(b)))
            }
        }
        _ => None,
    }
}

enum Flow {
    Next,
    Done,
    Untraced,
    Reload,
}

#[derive(Clone, Copy)]
struct Entry {
    closure: Ref<Closure>,
    base: usize,
    pc: usize,
}

enum Switched {
    Landed(Entry),
    Left,
    Not,
}

enum TopEntry {
    Lua(Entry),
    Leave(bool),
    Coroutine,
}

#[derive(Clone, Copy)]
struct Activation {
    closure: Ref<Closure>,
    base: usize,
    proto: Ref<Proto>,
}

impl Activation {
    #[inline(always)]
    fn of(window: &Window) -> Self {
        Self {
            closure: window.closure(),
            base: window.base(),
            proto: window.proto(),
        }
    }
}

#[derive(Clone, Copy)]
struct Arithmetic {
    op: ArithOp,
    left: Value,
    right: Value,
    flip: bool,
    operands: Operands,
}

#[derive(Clone, Copy)]
struct Site<'w> {
    activation: Activation,
    pc: usize,
    window: Passed<'w>,
    borrow: PhantomData<&'w Window>,
}

impl<'w> Site<'w> {
    #[inline(always)]
    fn at(window: &'w Window, pc: usize) -> Self {
        Site {
            activation: Activation::of(window),
            pc,
            window: window.pass(),
            borrow: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
struct Access {
    object: Value,
    key: Value,
    operand: Operand,
}

#[derive(Clone, Copy)]
struct Store {
    access: Access,
    value: Value,
}

impl Vm {
    /// Follows LuaJIT's `lj_dispatch_update`; the hook-free and hooked loops swap when the hook state changes.
    pub(crate) fn execute(&mut self, scope: RunScope) -> Result<(), LuaError> {
        let mut skip_trace = false;
        loop {
            let switch = if self.tracing {
                self.run_loop::<false, true, true>(scope, skip_trace)?
            } else if self.charges() {
                self.run_loop::<false, false, true>(scope, skip_trace)?
            } else {
                self.run_loop::<false, false, false>(scope, skip_trace)?
            };
            let Some(skip) = switch else {
                return Ok(());
            };
            skip_trace = skip;
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn run_one(&mut self, scope: RunScope) -> Result<(), LuaError> {
        if self.tracing {
            self.run_loop::<true, true, true>(scope, false).map(|_| ())
        } else {
            self.run_loop::<true, false, true>(scope, false).map(|_| ())
        }
    }

    /// `CHARGES` is false only for the interpreter tier, whose loop makes no compile-budget tests.
    fn run_loop<const STEP: bool, const HOOKS: bool, const CHARGES: bool>(&mut self, scope: RunScope, skip_trace: bool) -> Result<Option<bool>, LuaError> {
        self.dispatch::<STEP, HOOKS, CHARGES>(&scope, skip_trace)
    }

    /// Owns nothing that needs dropping, so debug builds give its calls no unwind cleanup or spill slots.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn dispatch<const STEP: bool, const HOOKS: bool, const CHARGES: bool>(&mut self, scope: &RunScope, skip_trace: bool) -> Result<Option<bool>, LuaError> {
        let mut skip_first = skip_trace;
        let mut record = Record::polling(&self.requests);
        loop {
            let Entry { closure, base, pc: resume_pc } = match self.top_entry(scope, STEP, HOOKS) {
                TopEntry::Lua(entry) => entry,
                TopEntry::Leave(switched) => return Ok(switched.then_some(skip_first)),
                TopEntry::Coroutine => {
                    let Some(leave) = self.coroutine_exit() else {
                        return Ok(None);
                    };
                    ok!(self.leave_coroutine(leave));
                    ok!(self.call_loop(None));
                    continue;
                }
            };
            let mut window = Window::enter(self, &raw mut record);
            if !STEP && CHARGES && self.charges() && window.charge_spent(crate::jit::ENTRY_CHARGE) && ok!(self.jit_budget_spent(closure, base, window.proto(), resume_pc, window.registers())) {
                continue;
            }
            let mut ip = window.ip_at(resume_pc);
            let mut skip_trace = std::mem::take(&mut skip_first);
            let mut stepped = false;
            ok!(self.safe_point(resume_pc));
            loop {
                if STEP {
                    if stepped {
                        window.save_pc(window.pc(ip));
                        return Ok(None);
                    }
                    stepped = true;
                }
                if HOOKS && self.tracing {
                    let pc = window.pc(ip);
                    ok!(self.trace_point(&mut window, pc, !skip_trace));
                }
                skip_trace = false;
                #[cfg(debug_assertions)]
                window.assert_current(self);
                let word = window.fetch(ip);
                let opcode = opcode_of(word);
                #[cfg(feature = "opcode-counts")]
                self.opcode_counts.record(opcode.generic());
                let flow = match opcode {
                    Dispatch::Move => {
                        decode!(word, Move { a, b });
                        window.set_register(a, window.register(b));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadI => {
                        decode!(word, LoadI { a, sbx });
                        window.set_register(a, Value::Integer(i64::from(sbx)));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadF => {
                        decode!(word, LoadF { a, sbx });
                        window.set_register(a, Value::Float(f64::from(sbx)));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadK => {
                        decode!(word, LoadK { a, bx });
                        window.set_register(a, quick::constant(window.pass(), bx));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadKx => {
                        ip = ip.add(1);
                        decode!(word, LoadKx { a });
                        let k = ok!(self.extra_argument(window.pass(), window.pc(ip)));
                        ip = ip.add(1);
                        window.set_register(a, quick::constant(window.pass(), k));
                        continue;
                    }
                    Dispatch::LoadFalse => {
                        decode!(word, LoadFalse { a });
                        window.set_register(a, Value::Boolean(false));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LFalseSkip => {
                        ip = ip.add(1);
                        decode!(word, LFalseSkip { a });
                        window.set_register(a, Value::Boolean(false));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadTrue => {
                        decode!(word, LoadTrue { a });
                        window.set_register(a, Value::Boolean(true));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::LoadNil => {
                        decode!(word, LoadNil { a, b });
                        let first = usize::from(a);
                        for register in first..=first + usize::from(b) {
                            window.set_register(register, Value::Nil);
                        }
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::GetUpval => {
                        decode!(word, GetUpval { a, b });
                        let value = window.upvalue_value(self, b);
                        window.set_register(a, value);
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::SetUpval => {
                        decode!(word, SetUpval { a, b });
                        let value = window.register(a);
                        window.set_upvalue(self, b, value);
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::GetTabUp => {
                        decode!(word, GetTabUp { a, b, c });
                        let object = window.upvalue_value(self, b);
                        let key = quick::constant(window.pass(), u32::from(c));
                        let value = self.field_value_in(|| window.feedback_at(ip), object, key);
                        if !value.is_nil() {
                            window.copy_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.tab_up_get(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetTabUpShape => {
                        decode!(word, GetTabUp { a, b, c });
                        let object = window.upvalue_value(self, b);
                        let value = self.shape_hit(window.feedback_at(ip), object);
                        if !value.is_nil() {
                            window.copy_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.tab_up_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetTable => {
                        decode!(word, GetTable { a, b, c });
                        let (object, key) = (window.register(b), window.register(c));
                        let value = quick::int_get(self, object, key);
                        if !value.is_nil() {
                            window.rewrite(window.pc(ip), Dispatch::GetTableInt);
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        match self.index_resolved(object, key) {
                            Some(value) => {
                                window.set_register(a, value);
                                ip = ip.add(1);
                                continue;
                            }
                            None => { ip = ip.add(1); ok!(self.index_slow(Activation::of(&window), window.pc(ip), a, Access { object, key, operand: Operand::Register(b) })) },
                        }
                    }
                    Dispatch::GetTableAny => {
                        decode!(word, GetTable { a, b, c });
                        let (object, key) = (window.register(b), window.register(c));
                        match self.index_resolved(object, key) {
                            Some(value) => {
                                window.set_register(a, value);
                                ip = ip.add(1);
                                continue;
                            }
                            None => { ip = ip.add(1); ok!(self.index_slow(Activation::of(&window), window.pc(ip), a, Access { object, key, operand: Operand::Register(b) })) },
                        }
                    }
                    Dispatch::GetTableInt => {
                        decode!(word, GetTable { a, b, c });
                        let value = quick::int_get_ref(self, window.register_ref(b), window.register_ref(c));
                        if !value.is_nil() {
                            window.copy_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.index_missed(Site::at(&window, window.pc(ip)), a, b, window.register(c), Dispatch::GetTableAny)) }
                    }
                    Dispatch::GetI => {
                        decode!(word, GetI { a, b, c });
                        let (object, key) = (window.register(b), Value::Integer(i64::from(c)));
                        let value = quick::int_get(self, object, key);
                        if !value.is_nil() {
                            window.rewrite(window.pc(ip), Dispatch::GetIInt);
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        match self.index_resolved(object, key) {
                            Some(value) => {
                                window.set_register(a, value);
                                ip = ip.add(1);
                                continue;
                            }
                            None => { ip = ip.add(1); ok!(self.index_slow(Activation::of(&window), window.pc(ip), a, Access { object, key, operand: Operand::Register(b) })) },
                        }
                    }
                    Dispatch::GetIAny => {
                        decode!(word, GetI { a, b, c });
                        let (object, key) = (window.register(b), Value::Integer(i64::from(c)));
                        match self.index_resolved(object, key) {
                            Some(value) => {
                                window.set_register(a, value);
                                ip = ip.add(1);
                                continue;
                            }
                            None => { ip = ip.add(1); ok!(self.index_slow(Activation::of(&window), window.pc(ip), a, Access { object, key, operand: Operand::Register(b) })) },
                        }
                    }
                    Dispatch::GetIInt => {
                        decode!(word, GetI { a, b, c });
                        let value = quick::int_get_ref(self, window.register_ref(b), &Value::Integer(i64::from(c)));
                        if !value.is_nil() {
                            window.copy_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.index_missed(Site::at(&window, window.pc(ip)), a, b, Value::Integer(i64::from(c)), Dispatch::GetIAny)) }
                    }
                    Dispatch::GetField => {
                        decode!(word, GetField { a, b, c });
                        let object = window.register(b);
                        let key = quick::constant(window.pass(), u32::from(c));
                        let probe = self.field_probe_in(window.feedback_at(ip), object, key);
                        if let Probe::Value(value) = probe {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        if let Probe::Absent(table) = probe
                            && self.absent_is_nil(table)
                        {
                            window.set_register(a, Value::Nil);
                            ip = ip.add(1);
                            continue;
                        }
                        if !self.heap.table_is_shaped(object)
                            && let Some(value) = self.index_resolved(object, key)
                        {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_get(Site::at(&window, window.pc(ip)), a, b, c, probe)) }
                    }
                    Dispatch::GetFieldShape => {
                        decode!(word, GetField { a, b, c });
                        let object = window.register(b);
                        let value = self.shape_hit(window.feedback_at(ip), object);
                        if !value.is_nil() {
                            window.copy_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetFieldShapes => {
                        decode!(word, GetField { a, b, c });
                        let object = window.register(b);
                        let value = self.shapes_hit(window.feedback_at(ip), object);
                        if !value.is_nil() {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetFieldAbsent => {
                        decode!(word, GetField { a, b, c });
                        let object = window.register(b);
                        if self.absent_hit(window.feedback_at(ip), object) {
                            window.set_register(a, Value::Nil);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetFieldMethod => {
                        decode!(word, GetField { a, b, c });
                        let object = window.register(b);
                        if let Value::Table(table) = object
                            && let Some(value) = self.method_hit_in(window.feedback_at(ip), table)
                        {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::GetVarg => {
                        ip = ip.add(1);
                        decode!(word, GetVarg { a, c, .. });
                        self.get_vararg(Activation::of(&window), a, c);
                        Flow::Next
                    }
                    Dispatch::SelfOp => {
                        decode!(word, SelfOp { a, b, c });
                        let object = window.register(b);
                        let key = quick::constant(window.pass(), u32::from(c));
                        let probe = self.field_probe_in(window.feedback_at(ip), object, key);
                        if let Probe::Value(method) = probe {
                            window.set_register(a + 1, object);
                            window.set_register(a, method);
                            ip = ip.add(1);
                            continue;
                        }
                        if let Value::Table(table) = object
                            && let Some(method) = self.method_hit_in(window.feedback_at(ip), table)
                        {
                            window.set_register(a + 1, object);
                            window.set_register(a, method);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.method_get(Site::at(&window, window.pc(ip)), a, b, c, probe)) }
                    }
                    Dispatch::SelfShape => {
                        decode!(word, SelfOp { a, b, c });
                        let object = window.register(b);
                        let method = self.shape_hit(window.feedback_at(ip), object);
                        if !method.is_nil() {
                            window.move_register(b, a + 1);
                            window.copy_register(a, method);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.method_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::SelfMethod => {
                        decode!(word, SelfOp { a, b, c });
                        let object = window.register(b);
                        if let Value::Table(table) = object
                            && let Some(method) = self.method_hit_in(window.feedback_at(ip), table)
                        {
                            window.set_register(a + 1, object);
                            window.set_register(a, method);
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.method_get_missed(Site::at(&window, window.pc(ip)), a, b, c)) }
                    }
                    Dispatch::SetTabUp => {
                        ip = ip.add(1);
                        decode!(word, SetTabUp { a, b, c, k });
                        ok!(self.tab_up_set(Site::at(&window, window.pc(ip)), a, b, c, k))
                    }
                    Dispatch::SetTable => {
                        decode!(word, SetTable { a, b, c, k });
                        if quick::int_set(self, window.register_ref(a), window.register_ref(b), quick::rk_ref(&window, c, k)) {
                            window.rewrite(window.pc(ip), Dispatch::SetTableInt);
                            ip = ip.add(1);
                            continue;
                        }
                        let store = Store {
                            access: Access {
                                object: window.register(a),
                                key: window.register(b),
                                operand: Operand::Register(a),
                            },
                            value: quick::rk(window.pass(), c, k),
                        };
                        if quick::raw_set(self, store.access.object, store.access.key, store.value) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_slow(window.pc(ip), store)) }
                    }
                    Dispatch::SetTableAny => {
                        decode!(word, SetTable { a, b, c, k });
                        let store = Store {
                            access: Access {
                                object: window.register(a),
                                key: window.register(b),
                                operand: Operand::Register(a),
                            },
                            value: quick::rk(window.pass(), c, k),
                        };
                        if quick::raw_set(self, store.access.object, store.access.key, store.value) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_slow(window.pc(ip), store)) }
                    }
                    Dispatch::SetTableInt => {
                        decode!(word, SetTable { a, b, c, k });
                        if quick::int_set(self, window.register_ref(a), window.register_ref(b), quick::rk_ref(&window, c, k)) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_missed(window.pass(), window.pc(ip), (a, c, k), window.register(b), Dispatch::SetTableAny)) }
                    }
                    Dispatch::SetI => {
                        decode!(word, SetI { a, b, c, k });
                        if quick::int_set(self, window.register_ref(a), &Value::Integer(i64::from(b)), quick::rk_ref(&window, c, k)) {
                            window.rewrite(window.pc(ip), Dispatch::SetIInt);
                            ip = ip.add(1);
                            continue;
                        }
                        let store = Store {
                            access: Access {
                                object: window.register(a),
                                key: Value::Integer(i64::from(b)),
                                operand: Operand::Register(a),
                            },
                            value: quick::rk(window.pass(), c, k),
                        };
                        if quick::raw_set(self, store.access.object, store.access.key, store.value) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_slow(window.pc(ip), store)) }
                    }
                    Dispatch::SetIAny => {
                        decode!(word, SetI { a, b, c, k });
                        let store = Store {
                            access: Access {
                                object: window.register(a),
                                key: Value::Integer(i64::from(b)),
                                operand: Operand::Register(a),
                            },
                            value: quick::rk(window.pass(), c, k),
                        };
                        if quick::raw_set(self, store.access.object, store.access.key, store.value) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_slow(window.pc(ip), store)) }
                    }
                    Dispatch::SetIInt => {
                        decode!(word, SetI { a, b, c, k });
                        if quick::int_set(self, window.register_ref(a), &Value::Integer(i64::from(b)), quick::rk_ref(&window, c, k)) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.set_missed(window.pass(), window.pc(ip), (a, c, k), Value::Integer(i64::from(b)), Dispatch::SetIAny)) }
                    }
                    Dispatch::SetField => {
                        decode!(word, SetField { a, b, c, k });
                        let object = window.register(a);
                        let key = quick::constant(window.pass(), u32::from(b));
                        let value = quick::rk(window.pass(), c, k);
                        if self.field_store_hit_in(window.feedback_at(ip), object, key, value) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_set(Site::at(&window, window.pc(ip)), a, b, c, k)) }
                    }
                    Dispatch::SetFieldShape => {
                        decode!(word, SetField { a, b, c, k });
                        if self.shape_store_hit(window.feedback_at(ip), window.register(a), quick::rk(window.pass(), c, k)) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_set_missed(Site::at(&window, window.pc(ip)), a, b, c, k)) }
                    }
                    Dispatch::SetFieldAdd => {
                        decode!(word, SetField { a, b, c, k });
                        if self.transition_hit(window.feedback_at(ip), window.register(a), quick::rk(window.pass(), c, k)) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.field_set_missed(Site::at(&window, window.pc(ip)), a, b, c, k)) }
                    }
                    Dispatch::SetTabUpShape => {
                        decode!(word, SetTabUp { a, b, c, k });
                        let object = window.upvalue_value(self, a);
                        if self.shape_store_hit(window.feedback_at(ip), object, quick::rk(window.pass(), c, k)) {
                            ip = ip.add(1);
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.tab_up_set_missed(Site::at(&window, window.pc(ip)), a, b, c, k)) }
                    }
                    Dispatch::NewTable | Dispatch::NewTableOpen => {
                        ip = ip.add(1);
                        let instruction = crate::encoding::operands(Opcode::NewTable, word);
                        let Instruction::NewTable { a, b, c, k } = instruction else {
                            unreachable!()
                        };
                        let extra = if k { ok!(self.extra_argument(window.pass(), window.pc(ip))) } else { 0 };
                        let array = u64::from(c) + u64::from(extra) * u64::from(SETLIST_BLOCK);
                        ip = ip.add(1);
                        if ok!(self.allocation_point(|| Activation::of(&window), window.pc(ip), instruction)) {
                            window.refresh(self);
                        }
                        let table = ok!(self.new_table_sized(array, b, opcode == Dispatch::NewTableOpen));
                        if self.options.memory_limit.is_some() {
                            window.refresh(self);
                        }
                        window.set_register(a, Value::Table(table));
                        continue;
                    }
                    Dispatch::Closure => {
                        ip = ip.add(1);
                        let instruction = crate::encoding::operands(Opcode::Closure, word);
                        let Instruction::Closure { a, bx } = instruction else {
                            unreachable!()
                        };
                        if ok!(self.allocation_point(|| Activation::of(&window), window.pc(ip), instruction)) {
                            window.refresh(self);
                        }
                        let closure = self.new_closure(window.pass(), bx as usize);
                        window.set_register(a, Value::Closure(closure));
                        continue;
                    }
                    Dispatch::Add => {
                        decode!(word, Add { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Add, window.register(b), window.register(c))
                    }
                    Dispatch::Sub => {
                        decode!(word, Sub { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Sub, window.register(b), window.register(c))
                    }
                    Dispatch::Mul => {
                        decode!(word, Mul { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Mul, window.register(b), window.register(c))
                    }
                    Dispatch::Mod => {
                        decode!(word, Mod { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Mod, window.register(b), window.register(c))
                    }
                    Dispatch::Pow => {
                        decode!(word, Pow { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Pow, window.register(b), window.register(c))
                    }
                    Dispatch::Div => {
                        decode!(word, Div { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Div, window.register(b), window.register(c))
                    }
                    Dispatch::IDiv => {
                        decode!(word, IDiv { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::IDiv, window.register(b), window.register(c))
                    }
                    Dispatch::BAnd => {
                        decode!(word, BAnd { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BAnd, window.register(b), window.register(c))
                    }
                    Dispatch::BOr => {
                        decode!(word, BOr { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BOr, window.register(b), window.register(c))
                    }
                    Dispatch::BXor => {
                        decode!(word, BXor { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BXor, window.register(b), window.register(c))
                    }
                    Dispatch::Shl => {
                        decode!(word, Shl { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Shl, window.register(b), window.register(c))
                    }
                    Dispatch::Shr => {
                        decode!(word, Shr { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Shr, window.register(b), window.register(c))
                    }
                    Dispatch::AddK => {
                        decode!(word, AddK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Add, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::SubK => {
                        decode!(word, SubK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Sub, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::MulK => {
                        decode!(word, MulK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Mul, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::ModK => {
                        decode!(word, ModK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Mod, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::PowK => {
                        decode!(word, PowK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Pow, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::DivK => {
                        decode!(word, DivK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::Div, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::IDivK => {
                        decode!(word, IDivK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::IDiv, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::BAndK => {
                        decode!(word, BAndK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BAnd, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::BOrK => {
                        decode!(word, BOrK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BOr, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::BXorK => {
                        decode!(word, BXorK { a, b, c });
                        fast_arith!(window, ip, a, ArithOp::BXor, window.register(b), quick::constant(window.pass(), u32::from(c)))
                    }
                    Dispatch::AddI => {
                        decode!(word, AddI { a, b, sc });
                        fast_arith!(window, ip, a, ArithOp::Add, window.register(b), Value::Integer(i64::from(sc)))
                    }
                    Dispatch::ShrI => {
                        decode!(word, ShrI { a, b, sc });
                        fast_arith!(window, ip, a, ArithOp::Shr, window.register(b), Value::Integer(i64::from(sc)))
                    }
                    Dispatch::ShlI => {
                        decode!(word, ShlI { a, b, sc });
                        fast_arith!(window, ip, a, ArithOp::Shl, Value::Integer(i64::from(sc)), window.register(b))
                    }
                    Dispatch::MmBin => {
                        ip = ip.add(1);
                        decode!(word, MmBin { a, b, c });
                        let (target, op) = ok!(self.arithmetic_site(window.pass(), window.pc(ip), c));
                        let operands = Operands {
                            left: Operand::Register(a),
                            right: Operand::Register(b),
                        };
                        let arithmetic = Arithmetic {
                            op,
                            left: window.register(a),
                            right: window.register(b),
                            flip: false,
                            operands,
                        };
                        ok!(self.arith_slow(Activation::of(&window), window.pc(ip), target, arithmetic))
                    }
                    Dispatch::MmBinI => {
                        ip = ip.add(1);
                        decode!(word, MmBinI { a, sb, c, k });
                        let (target, op) = ok!(self.arithmetic_site(window.pass(), window.pc(ip), c));
                        let operands = Operands {
                            left: Operand::Register(a),
                            right: Operand::Constant,
                        };
                        let arithmetic = Arithmetic {
                            op,
                            left: window.register(a),
                            right: Value::Integer(i64::from(sb)),
                            flip: k,
                            operands,
                        };
                        ok!(self.arith_slow(Activation::of(&window), window.pc(ip), target, arithmetic))
                    }
                    Dispatch::MmBinK => {
                        ip = ip.add(1);
                        decode!(word, MmBinK { a, b, c, k });
                        let (target, op) = ok!(self.arithmetic_site(window.pass(), window.pc(ip), c));
                        let operands = Operands {
                            left: Operand::Register(a),
                            right: Operand::Constant,
                        };
                        let arithmetic = Arithmetic {
                            op,
                            left: window.register(a),
                            right: quick::constant(window.pass(), u32::from(b)),
                            flip: k,
                            operands,
                        };
                        ok!(self.arith_slow(Activation::of(&window), window.pc(ip), target, arithmetic))
                    }
                    Dispatch::Unm => {
                        decode!(word, Unm { a, b });
                        match fast::unary(UnaryOp::Minus, window.register(b)) {
                        Some(value) => {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        None => { ip = ip.add(1); ok!(self.unary_slow(Activation::of(&window), window.pc(ip), a, UnaryOp::Minus, b)) },
                    }
                    }
                    Dispatch::BNot => {
                        decode!(word, BNot { a, b });
                        match fast::unary(UnaryOp::BNot, window.register(b)) {
                        Some(value) => {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        None => { ip = ip.add(1); ok!(self.unary_slow(Activation::of(&window), window.pc(ip), a, UnaryOp::BNot, b)) },
                    }
                    }
                    Dispatch::Not => {
                        decode!(word, Not { a, b });
                        window.set_register(a, Value::Boolean(window.register(b).is_falsy()));
                        ip = ip.add(1);
                        continue;
                    }
                    Dispatch::Len => {
                        decode!(word, Len { a, b });
                        match self.raw_length(window.register(b)) {
                        Some(value) => {
                            window.set_register(a, value);
                            ip = ip.add(1);
                            continue;
                        }
                        None => { ip = ip.add(1); ok!(self.unary_slow(Activation::of(&window), window.pc(ip), a, UnaryOp::Len, b)) },
                    }
                    }
                    Dispatch::Jmp => {
                        ip = ip.add(1);
                        decode!(word, Jmp { sj });
                        ip = ip.offset(sj as isize);
                        if sj < 0 {
                            ok!(self.safe_point(window.pc(ip)));
                            if !STEP && CHARGES && self.charges() && window.charge_spent(sj.unsigned_abs()) {
                                window.save_pc(window.pc(ip));
                                break;
                            }
                        }
                        continue;
                    }
                    Dispatch::Test => {
                        decode!(word, Test { a, k });
                        quick::conditional_jump(window.pass(), &mut ip, !window.register(a).is_falsy(), k);
                        continue;
                    }
                    Dispatch::TestSet => {
                        decode!(word, TestSet { a, b, k });
                        let value = window.register(b);
                        if value.is_falsy() == k {
                            ip = ip.add(2);
                        } else {
                            window.set_register(a, value);
                            quick::take_jump(window.pass(), &mut ip);
                        }
                        continue;
                    }
                    Dispatch::EqK => {
                        decode!(word, EqK { a, b, k });
                        let (left, right) = (window.register(a), quick::constant(window.pass(), u32::from(b)));
                        let condition = match (left, right) {
                            (Value::Integer(left), Value::Integer(right)) => left == right,
                            _ => self.heap.raw_equal(left, right),
                        };
                        quick::conditional_jump(window.pass(), &mut ip, condition, k);
                        continue;
                    }
                    Dispatch::EqI => {
                        decode!(word, EqI { a, sb, k, .. });
                        let condition = match window.register(a) {
                            Value::Integer(value) => value == i64::from(sb),
                            Value::Float(value) => value == f64::from(sb),
                            _ => false,
                        };
                        quick::conditional_jump(window.pass(), &mut ip, condition, k);
                        continue;
                    }
                    Dispatch::Eq => {
                        decode!(word, Eq { a, b, k });
                        let (left, right) = (window.register(a), window.register(b));
                        match fast::equal(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.equal_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::Lt => {
                        decode!(word, Lt { a, b, k });
                        if let (Value::Integer(left), Value::Integer(right)) = (window.register_ref(a), window.register_ref(b)) {
                            quick::conditional_jump(window.pass(), &mut ip, *left < *right, k);
                            continue;
                        }
                        let (left, right) = (window.register(a), window.register(b));
                        match fast::less_than(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_than_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::Le => {
                        decode!(word, Le { a, b, k });
                        if let (Value::Integer(left), Value::Integer(right)) = (window.register_ref(a), window.register_ref(b)) {
                            quick::conditional_jump(window.pass(), &mut ip, *left <= *right, k);
                            continue;
                        }
                        let (left, right) = (window.register(a), window.register(b));
                        match fast::less_equal(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_equal_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::LtI => {
                        decode!(word, LtI { a, sb, c, k });
                        if let Some(outcome) = fast::less_than_immediate(window.register_ref(a), sb) {
                            quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                            continue;
                        }
                        let (left, right) = (window.register(a), fast::immediate(sb, c != 0));
                        match fast::less_than(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_than_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::LeI => {
                        decode!(word, LeI { a, sb, c, k });
                        if let Some(outcome) = fast::less_equal_immediate(window.register_ref(a), sb) {
                            quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                            continue;
                        }
                        let (left, right) = (window.register(a), fast::immediate(sb, c != 0));
                        match fast::less_equal(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_equal_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::GtI => {
                        decode!(word, GtI { a, sb, c, k });
                        if let Some(outcome) = fast::greater_than_immediate(window.register_ref(a), sb) {
                            quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                            continue;
                        }
                        let (left, right) = (fast::immediate(sb, c != 0), window.register(a));
                        match fast::less_than(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_than_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::GeI => {
                        decode!(word, GeI { a, sb, c, k });
                        if let Some(outcome) = fast::greater_equal_immediate(window.register_ref(a), sb) {
                            quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                            continue;
                        }
                        let (left, right) = (fast::immediate(sb, c != 0), window.register(a));
                        match fast::less_equal(left, right) {
                            Some(outcome) => {
                                quick::conditional_jump(window.pass(), &mut ip, outcome, k);
                                continue;
                            }
                            None => { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.less_equal_slow(Activation::of(&window), &mut moved, left, right, k)); ip = window.ip_at(moved); flow } },
                        }
                    }
                    Dispatch::ForLoop => {
                        ip = ip.add(1);
                        decode!(word, ForLoop { a, bx });
                        match quick::for_loop(window.pass(), usize::from(a)) {
                            Some(true) => {
                                ip = ip.sub(bx as usize);
                                if window.requested() && ok!(self.loop_back_edge(window.pass(), window.pc(ip))) && !STEP {
                                    window.save_pc(window.pc(ip));
                                    break;
                                }
                            }
                            Some(false) => {}
                            None => return Err(self.runtime_error(b"invalid instruction")),
                        }
                        continue;
                    }
                    Dispatch::ForPrep => {
                        ip = ip.add(1);
                        decode!(word, ForPrep { a, bx });
                        match quick::for_prep_integers(window.pass(), usize::from(a)) {
                        Some(skip) => {
                            if skip {
                                ip = ip.add(bx as usize + 1);
                            } else if !STEP && self.loop_entry_spends(window.pass(), window.proto(), window.pc(ip), a, bx) {
                                window.save_pc(window.pc(ip));
                                break;
                            }
                            continue;
                        }
                        None => {
                            let start = window.pc(ip);
                            { let mut moved = window.pc(ip); let flow = ok!(self.for_prep_slow(Activation::of(&window), &mut moved, a, bx)); ip = window.ip_at(moved); flow };
                            window.refresh(self);
                            if !STEP && CHARGES && window.pc(ip) == start && self.charges() && window.charge_spent(quick::loop_charge(window.pass(), usize::from(a), bx)) {
                                window.save_pc(window.pc(ip));
                                break;
                            }
                            continue;
                        }
                    }
                    }
                    Dispatch::TForLoop => {
                        ip = ip.add(1);
                        decode!(word, TForLoop { a, bx });
                        let control = window.register(usize::from(a) + 3);
                        if !control.is_nil() {
                            ip = ip.sub(bx as usize);
                            ok!(self.safe_point(window.pc(ip)));
                            if !STEP && CHARGES && self.charges() && window.charge_spent(bx) {
                                window.save_pc(window.pc(ip));
                                break;
                            }
                        }
                        continue;
                    }
                    Dispatch::SetList => {
                        ip = ip.add(1);
                        decode!(word, SetList { a, b, c, k });
                        let mut stored = c;
                        if k {
                            let extra = ok!(self.extra_argument(window.pass(), window.pc(ip)));
                            let last = u64::from(stored) + u64::from(extra) * u64::from(SETLIST_BLOCK);
                            if last > MAX_ARRAY_SIZE {
                                return Err(self.memory_error());
                            }
                            stored = last as u32;
                            ip = ip.add(1);
                        }
                        ok!(self.set_list(Activation::of(&window), a, b, stored));
                        window.refresh(self);
                        continue;
                    }
                    Dispatch::VarargPrep => {
                        ip = ip.add(1);
                        decode!(word, VarargPrep { .. });
                        window = ok!(self.prepare_varargs(&window));
                        if HOOKS && self.tracing {
                            ok!(self.hook_call_after_prologue());
                            window.refresh(self);
                        }
                        continue;
                    }
                    Dispatch::Concat => {
                        ip = ip.add(1);
                        let instruction = crate::encoding::operands(Opcode::Concat, word);
                        let Instruction::Concat { a, b } = instruction else {
                            unreachable!()
                        };
                        if ok!(self.allocation_point(|| Activation::of(&window), window.pc(ip), instruction)) {
                            window.refresh(self);
                        }
                        if b == 2
                            && matches!(window.register(a + 1), Value::String(_))
                            && let Some(value) = self.concat_short_strings(window.register(a), window.register(a + 1))
                        {
                            window.set_register(a, value);
                            continue;
                        }
                        window.save_pc(window.pc(ip));
                        if ok!(self.concat_loop(window.base(), a, usize::from(b))) {
                            if self.options.memory_limit.is_some() {
                                window.refresh(self);
                            }
                            continue;
                        }
                        Flow::Reload
                    }
                    Dispatch::Close => {
                        ip = ip.add(1);
                        decode!(word, Close { a });
                        { let mut moved = window.pc(ip); let flow = ok!(self.close_slow(Activation::of(&window), &mut moved, a)); ip = window.ip_at(moved); flow }
                    }
                    Dispatch::Tbc => {
                        ip = ip.add(1);
                        decode!(word, Tbc { a });
                        ok!(self.tbc_slow(window.pc(ip), a))
                    }
                    Dispatch::ErrNil => {
                        ip = ip.add(1);
                        decode!(word, ErrNil { a, bx });
                        ok!(self.err_nil(Activation::of(&window), window.pc(ip), a, bx))
                    }
                    Dispatch::Call => {
                        decode!(word, Call { a, b, c });
                        match window.register(a) {
                        Value::Closure(closure) if !HOOKS && !STEP => {
                            let next = ok!(self.enter_called::<CHARGES>(&window, window.next_pc(ip), closure, a, b, c));
                            if CHARGES && self.charges() && next.charge_spent(crate::jit::ENTRY_CHARGE) && ok!(self.jit_budget_spent(closure, next.base(), next.proto(), 0, next.registers())) {
                                break;
                            }
                            window = next;
                            ip = window.ip_at(0);
                            ok!(self.safe_point(0));
                            continue;
                        }
                        Value::Closure(closure) if !HOOKS || !self.tracing => {
                            { ip = ip.add(1); ok!(self.call_closure(Activation::of(&window), window.pc(ip), closure, a, b, c)); }
                            break;
                        }
                        Value::Native(native) => match self.builtin_of(native) {
                            Some(builtin @ (Builtin::Resume | Builtin::Wrap | Builtin::Yield)) if !HOOKS && !STEP => {
                                match ok!(self.switch_at(Activation::of(&window), window.next_pc(ip), native, builtin, (a, b, c), scope)) {
                                    Switched::Landed(Entry { pc: resume, .. }) => {
                                        let next = Window::enter(self, window.record());
                                        if CHARGES && self.charges() && next.charge_spent(crate::jit::ENTRY_CHARGE) && ok!(self.jit_budget_spent(next.closure(), next.base(), next.proto(), resume, next.registers())) {
                                            break;
                                        }
                                        window = next;
                                        ip = window.ip_at(resume);
                                        ok!(self.safe_point(window.pc(ip)));
                                        continue;
                                    }
                                    Switched::Left => break,
                                    Switched::Not => {
                                        ip = ip.add(1);
                                        ok!(self.call_native_at(Activation::of(&window), window.pc(ip), native, a, b, c))
                                    }
                                }
                            }
                            Some(builtin) if (!HOOKS || !self.tracing) && c == 2 && b != 0 => {
                                if let Some(value) = self.builtin_value(builtin, window.base() + usize::from(a), usize::from(b) - 1) {
                                    window.set_register(a, value);
                                    ip = ip.add(1);
                                    continue;
                                }
                                { ip = ip.add(1); ok!(self.call_builtin_at(Activation::of(&window), window.pc(ip), native, builtin, (a, b, c))) }
                            }
                            Some(builtin) if !HOOKS || !self.tracing => { ip = ip.add(1); ok!(self.call_builtin_at(Activation::of(&window), window.pc(ip), native, builtin, (a, b, c))) },
                            _ => { ip = ip.add(1); ok!(self.call_native_at(Activation::of(&window), window.pc(ip), native, a, b, c)) },
                        },
                        _ => { ip = ip.add(1); ok!(self.call_slow(Activation::of(&window), window.pc(ip), a, b, c)) },
                    }
                    }
                    Dispatch::TailCall => {
                        decode!(word, TailCall { a, b, k, .. });
                        if !HOOKS
                            && !STEP
                            && let Value::Closure(closure) = window.register(a)
                            && let Some(next) = self.tail_called(&window, closure, a, b, k)
                        {
                            if CHARGES && self.charges() && next.charge_spent(crate::jit::ENTRY_CHARGE) && ok!(self.jit_budget_spent(closure, next.base(), next.proto(), 0, next.registers())) {
                                break;
                            }
                            window = next;
                            ip = window.ip_at(0);
                            ok!(self.safe_point(0));
                            continue;
                        }
                        { ip = ip.add(1); ok!(self.tail_call_slow(Activation::of(&window), window.pc(ip), a, b, k)) }
                    }
                    Dispatch::Return => {
                        decode!(word, Return { a, b, k, .. });
                        let first = window.base() + usize::from(a);
                        if !HOOKS
                            && !STEP
                            && let Some(count) = self.value_count(first, b)
                            && (!k || self.closed_for_return(window.base()))
                        {
                            return_inline!(self, window, ip, scope, CHARGES, first, count);
                        } else if !k && b != 0 && (!HOOKS || !self.tracing) && self.return_to_idle_caller(window.base() + usize::from(a), usize::from(b) - 1) {
                            break;
                        }
                        { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.return_slow(Activation::of(&window), &mut moved, a, b, k)); ip = window.ip_at(moved); flow } }
                    }
                    Dispatch::Return0 => {
                        decode!(word, Return0 { .. });
                        if !HOOKS && !STEP {
                            return_inline!(self, window, ip, scope, CHARGES, window.base(), 0);
                        } else if (!HOOKS || !self.tracing) && self.return_to_idle_caller(window.base(), 0) {
                            break;
                        }
                        { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.return_values(Activation::of(&window), window.base(), 0, false, &mut moved)); ip = window.ip_at(moved); flow } }
                    }
                    Dispatch::Return1 => {
                        decode!(word, Return1 { a });
                        let first = window.base() + usize::from(a);
                        if !HOOKS && !STEP {
                            return_inline!(self, window, ip, scope, CHARGES, first, 1);
                        } else if (!HOOKS || !self.tracing) && self.return_to_idle_caller(first, 1) {
                            break;
                        }
                        { ip = ip.add(1); { let mut moved = window.pc(ip); let flow = ok!(self.return_values(Activation::of(&window), first, 1, false, &mut moved)); ip = window.ip_at(moved); flow } }
                    }
                    Dispatch::TForPrep => {
                        ip = ip.add(1);
                        decode!(word, TForPrep { a, bx });
                        { let mut moved = window.pc(ip); let flow = ok!(self.tfor_prep_slow(Activation::of(&window), &mut moved, a, bx)); ip = window.ip_at(moved); flow }
                    }
                    Dispatch::TForCall => {
                        ip = ip.add(1);
                        decode!(word, TForCall { a, c });
                        ok!(self.generic_call(Activation::of(&window), window.pc(ip), a, c))
                    }
                    Dispatch::Vararg => {
                        ip = ip.add(1);
                        decode!(word, Vararg { a, b, c, k });
                        if !k && Self::copy_hidden_varargs(&window, a, c) {
                            continue;
                        }
                        ok!(self.copy_varargs(Activation::of(&window), a, c, k.then_some(b)));
                        Flow::Next
                    }
                    Dispatch::ExtraArg => {
                        decode!(word, ExtraArg { .. });
                        return Err(self.runtime_error(b"invalid instruction"))
                    }
                };
                match flow {
                    Flow::Done => {}
                    Flow::Next => window.refresh(self),
                    Flow::Untraced => {
                        window.refresh(self);
                        skip_trace = true;
                    }
                    Flow::Reload => break,
                }
                if !STEP && self.tracing != HOOKS {
                    window.save_pc(window.pc(ip));
                    return Ok(Some(skip_trace));
                }
            }
            if STEP {
                return Ok(None);
            }
        }
    }

    fn extra_argument(&mut self, window: Passed<'_>, pc: usize) -> Result<u32, LuaError> {
        match window.instruction(pc) {
            Instruction::ExtraArg { ax } => Ok(ax),
            _ => Err(self.runtime_error(b"invalid instruction")),
        }
    }

    fn arithmetic_site(&mut self, window: Passed<'_>, pc: usize, event: u8) -> Result<(u8, ArithOp), LuaError> {
        let target = pc.checked_sub(2).and_then(|index| arithmetic_target(window.instruction(index)));
        match (target, arith_of_event(event)) {
            (Some(target), Some(op)) => Ok((target, op)),
            _ => Err(self.runtime_error(b"invalid instruction")),
        }
    }

    /// Refuses before taking memory. An open constructor's array gets room for one more item, so its `SETLIST` does not reallocate.
    fn new_table_sized(&mut self, array: u64, hash_log: u8, open_list: bool) -> Result<Ref<Table>, LuaError> {
        if array > MAX_ARRAY_SIZE || hash_log > MAX_HASH_LOG {
            return Err(self.memory_error());
        }
        let hash = hash_size(hash_log);
        let spare = usize::from(array > 0 && open_list);
        self.reserve(Table::sized_bytes(array as usize + spare, hash))?;
        let table = Table::with_spare(array as usize, spare, hash).map_err(|_| self.memory_error())?;
        Ok(self.heap.alloc(table))
    }

    fn trace_point(&mut self, window: &mut Window, pc: usize, trace: bool) -> Result<(), LuaError> {
        self.save_pc(pc);
        if trace {
            self.trace_instruction(pc)?;
        }
        window.refresh(self);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn safe_point(&mut self, pc: usize) -> Result<(), LuaError> {
        if self.requests.load(Ordering::Relaxed) != 0 {
            return self.request_point(pc);
        }
        Ok(())
    }

    /// A charge that would spend the budget before a site has feedback is put off to the first back-edge, so compiled code sees one iteration's feedback.
    #[inline(always)]
    fn loop_entry_spends(&mut self, window: Passed<'_>, proto: Ref<Proto>, pc: usize, a: u8, bx: u32) -> bool {
        if !self.charges() {
            return false;
        }
        let charge = quick::loop_charge(window.pass(), usize::from(a), bx);
        if !window.would_spend(charge) {
            window.charge(charge);
            return false;
        }
        self.charge_or_defer(window.pass(), proto, pc, charge, bx)
    }

    #[inline(never)]
    fn charge_or_defer(&mut self, window: Passed<'_>, proto: Ref<Proto>, pc: usize, charge: u32, length: u32) -> bool {
        let body = pc..pc + length as usize;
        if self.heap.protos.live(proto).sites_lack_feedback(body) {
            self.deferred_loop_charge = Some(super::DeferredLoopCharge {
                proto,
                charge,
                thread: self.current_thread,
                depth: self.frame_count(),
                watch: false,
            });
            self.requests.fetch_or(LOOP_CHARGE, Ordering::Relaxed);
            return false;
        }
        window.charge_spent(charge)
    }

    #[inline(never)]
    fn loop_back_edge(&mut self, window: Passed<'_>, pc: usize) -> Result<bool, LuaError> {
        let deferred = self.deferred_loop_charge.is_some();
        self.request_point(pc)?;
        Ok(deferred && self.deferred_loop_charge.is_none() && window.charge_spent(0))
    }

    /// A request point in a callee leaves the charge pending; one after the frame is gone, or on another thread, drops it.
    fn take_deferred_loop_charge(&mut self) {
        if self.requests.load(Ordering::Relaxed) & LOOP_CHARGE == 0 {
            return;
        }
        let Some(pending) = self.deferred_loop_charge else {
            take_request(&self.requests, LOOP_CHARGE);
            return;
        };
        let depth = self.frame_count();
        if pending.thread == self.current_thread && depth > pending.depth {
            return;
        }
        take_request(&self.requests, LOOP_CHARGE);
        self.deferred_loop_charge = None;
        if pending.thread != self.current_thread || depth != pending.depth {
            return;
        }
        if self.charge_budget(pending.proto, pending.charge) && pending.watch {
            self.deferred_loop_charge = Some(pending);
            self.requests.fetch_or(LOOP_CHARGE, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn charges(&self) -> bool {
        cfg!(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64"))) && self.options.tier != crate::Tier::Interpreter
    }

    fn charge_budget(&self, proto: Ref<Proto>, charge: u32) -> bool {
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if let Some(function) = self.heap.protos.try_get(proto) {
            let budget = function.jit.budget();
            budget.set(budget.get().wrapping_sub(i64::from(charge)));
            return budget.get() > 0;
        }
        let _ = (proto, charge);
        false
    }

    /// Holds off the first pause until one request point has passed, so a host pausing before every resume still makes progress.
    #[inline(never)]
    fn request_point(&mut self, pc: usize) -> Result<(), LuaError> {
        self.save_pc(pc);
        self.take_deferred_loop_charge();
        if interrupt_requested(&self.requests) {
            return Err(self.interrupted());
        }
        let holdoff = std::mem::take(&mut self.pause_holdoff);
        if !holdoff && self.take_pause_request() {
            return Err(LuaError::pause());
        }
        Ok(())
    }

    /// Returns whether the collector ran; the caller then refreshes its window, since the collector may shrink the stack.
    #[inline(always)]
    fn allocation_point(&mut self, activation: impl FnOnce() -> Activation, pc: usize, instruction: Instruction) -> Result<bool, LuaError> {
        if self.heap.gc.debt > 0 {
            return Ok(false);
        }
        self.collect_at_allocation(activation(), pc, instruction)?;
        Ok(true)
    }

    #[inline(never)]
    fn collect_at_allocation(&mut self, activation: Activation, pc: usize, instruction: Instruction) -> Result<(), LuaError> {
        self.save_pc(pc);
        self.collect_before(activation, instruction);
        self.check_memory()
    }

    /// A full collection gets one chance to make room before the memory error.
    #[inline]
    pub(crate) fn reserve(&mut self, bytes: usize) -> Result<(), LuaError> {
        let Some(limit) = self.options.memory_limit else {
            return Ok(());
        };
        if self.fits_within(bytes, limit) {
            return Ok(());
        }
        self.reserve_collecting(bytes, limit)
    }

    #[cold]
    #[inline(never)]
    fn reserve_collecting(&mut self, bytes: usize, limit: usize) -> Result<(), LuaError> {
        self.gc_collect();
        if self.fits_within(bytes, limit) {
            return Ok(());
        }
        Err(self.memory_error())
    }

    fn fits_within(&self, bytes: usize, limit: usize) -> bool {
        i128::from(self.heap.gc.total) + bytes as i128 <= limit as i128
    }

    fn check_memory(&mut self) -> Result<(), LuaError> {
        let Some(limit) = self.options.memory_limit else {
            return Ok(());
        };
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        if self.heap.gc.total <= limit {
            return Ok(());
        }
        self.gc_collect();
        if self.heap.gc.total <= limit {
            return Ok(());
        }
        Err(self.memory_error())
    }

    fn collect_before(&mut self, activation: Activation, instruction: Instruction) {
        let Some(window) = gc_stack_window(instruction) else {
            return;
        };
        let top = match window {
            StackWindow::Registers(limit) => {
                let max_stack = usize::from(self.heap.protos.live(activation.proto).max_stack);
                activation.base.saturating_add(limit.min(max_stack))
            }
            StackWindow::Multret => self.thread().top,
        };
        self.gc_safe_point_at(activation.base, top);
    }

    fn arith_flipped(
        &mut self,
        op: ArithOp,
        left: Value,
        right: Value,
        operands: Operands,
        flip: bool,
    ) -> Result<Option<Value>, LuaError> {
        if !flip {
            return self.arith(op, left, right, operands);
        }
        let swapped = Operands {
            left: operands.right,
            right: operands.left,
        };
        self.arith(op, right, left, swapped)
    }

    fn reg(&self, activation: Activation, register: u8) -> Value {
        self.stack_get(activation.base + usize::from(register))
    }

    fn set_reg(&mut self, activation: Activation, register: u8, value: Value) {
        self.stack_set(activation.base + usize::from(register), value);
    }

    fn store(&mut self, activation: Activation, register: u8, value: Option<Value>) -> Flow {
        match value {
            Some(value) => {
                self.set_reg(activation, register, value);
                Flow::Next
            }
            None => Flow::Reload,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn upvalue_ref(&self, activation: Activation, index: u8) -> Ref<crate::object::UpVal> {
        self.heap.closures.live(activation.closure).upvalues[usize::from(index)].reference
    }

    fn take_jump(&self, activation: Activation, pc: &mut usize) {
        if let Instruction::Jmp { sj } = self.heap.protos.live(activation.proto).code[*pc] {
            *pc = (*pc as isize + 1 + sj as isize) as usize;
        }
    }

    fn conditional_jump(&self, activation: Activation, pc: &mut usize, condition: bool, expected: bool) {
        if condition != expected {
            *pc += 1;
        } else {
            self.take_jump(activation, pc);
        }
    }

    fn compare_jump(&self, activation: Activation, pc: &mut usize, outcome: Option<bool>, expected: bool) -> Flow {
        match outcome {
            Some(condition) => {
                self.conditional_jump(activation, pc, condition, expected);
                Flow::Next
            }
            None => Flow::Reload,
        }
    }

    #[inline(never)]
    fn field_get_missed(&mut self, site: Site<'_>, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        let window = site.window;
        window.miss(site.pc - 1, Dispatch::GetField);
        let probe = self.field_probe_in(window.feedback(site.pc - 1), window.register(b), quick::constant(window.pass(), u32::from(c)));
        self.field_get(site, a, b, c, probe)
    }

    #[inline(never)]
    fn method_get_missed(&mut self, site: Site<'_>, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        let window = site.window;
        window.miss(site.pc - 1, Dispatch::SelfOp);
        let probe = self.field_probe_in(window.feedback(site.pc - 1), window.register(b), quick::constant(window.pass(), u32::from(c)));
        self.method_get(site, a, b, c, probe)
    }

    #[inline(never)]
    fn tab_up_get_missed(&mut self, site: Site<'_>, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        site.window.miss(site.pc - 1, Dispatch::GetTabUp);
        self.tab_up_get(site, a, b, c)
    }

    #[inline(never)]
    fn field_set_missed(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, k: bool) -> Result<Flow, LuaError> {
        let window = site.window;
        if self.cached_store_in(window.feedback(site.pc - 1), window.register(a), quick::rk(window.pass(), c, k)) {
            return Ok(Flow::Done);
        }
        window.miss(site.pc - 1, Dispatch::SetField);
        let key = quick::constant(window.pass(), u32::from(b));
        if self.field_store_hit_in(window.feedback(site.pc - 1), window.register(a), key, quick::rk(window.pass(), c, k)) {
            return Ok(Flow::Done);
        }
        self.field_set(site, a, b, c, k)
    }

    #[inline(never)]
    fn tab_up_set_missed(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, k: bool) -> Result<Flow, LuaError> {
        site.window.miss(site.pc - 1, Dispatch::SetTabUp);
        self.tab_up_set(site, a, b, c, k)
    }

    #[inline(never)]
    fn index_missed(&mut self, site: Site<'_>, a: u8, b: u8, key: Value, generic: Dispatch) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        window.miss(pc - 1, generic);
        let access = Access {
            object: window.register(b),
            key,
            operand: Operand::Register(b),
        };
        match self.index_resolved(access.object, access.key) {
            Some(value) => {
                window.set_register(a, value);
                Ok(Flow::Done)
            }
            None => self.index_slow(activation, pc, a, access),
        }
    }

    #[inline(never)]
    fn set_missed(&mut self, window: Passed<'_>, pc: usize, (a, c, k): (u8, u8, bool), key: Value, generic: Dispatch) -> Result<Flow, LuaError> {
        window.miss(pc - 1, generic);
        let store = Store {
            access: Access {
                object: window.register(a),
                key,
                operand: Operand::Register(a),
            },
            value: quick::rk(window.pass(), c, k),
        };
        if quick::raw_set(self, store.access.object, store.access.key, store.value) {
            return Ok(Flow::Done);
        }
        self.set_slow(pc, store)
    }

    #[inline(never)]
    fn tab_up_get(&mut self, site: Site<'_>, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        let object = window.upvalue_value(self, b);
        let key = quick::constant(window.pass(), u32::from(c));
        match self.field_read_unprobed(activation.proto, pc - 1, object, key) {
            Some(value) => {
                window.set_register(a, value);
                Ok(Flow::Done)
            }
            None => self.index_slow(activation, pc, a, Access { object, key, operand: Operand::Upvalue(b) }),
        }
    }

    #[inline(never)]
    fn field_get(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, probe: Probe) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        let object = window.register(b);
        let key = quick::constant(window.pass(), u32::from(c));
        match self.field_read_probed(activation.proto, pc - 1, object, key, probe) {
            Some(value) => {
                window.set_register(a, value);
                Ok(Flow::Done)
            }
            None => self.index_slow(activation, pc, a, Access { object, key, operand: Operand::Register(b) }),
        }
    }

    #[inline(never)]
    fn method_get(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, probe: Probe) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        let object = window.register(b);
        let key = quick::constant(window.pass(), u32::from(c));
        match self.method_read_probed(activation.proto, pc - 1, object, key, probe) {
            Some(method) => {
                window.set_register(a + 1, object);
                window.set_register(a, method);
                Ok(Flow::Done)
            }
            None => self.self_slow(activation, pc, a, Access { object, key, operand: Operand::Register(b) }),
        }
    }

    #[inline(never)]
    fn tab_up_set(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, k: bool) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        let object = window.upvalue_value(self, a);
        let store = Store {
            access: Access {
                object,
                key: quick::constant(window.pass(), u32::from(b)),
                operand: Operand::Upvalue(a),
            },
            value: quick::rk(window.pass(), c, k),
        };
        if self.store_field(activation.proto, pc - 1, store.access.object, store.access.key, store.value) {
            return Ok(Flow::Done);
        }
        self.set_slow(pc, store)
    }

    #[inline(never)]
    fn field_set(&mut self, site: Site<'_>, a: u8, b: u8, c: u8, k: bool) -> Result<Flow, LuaError> {
        let Site { activation, pc, window, .. } = site;
        let store = Store {
            access: Access {
                object: window.register(a),
                key: quick::constant(window.pass(), u32::from(b)),
                operand: Operand::Register(a),
            },
            value: quick::rk(window.pass(), c, k),
        };
        if self.store_field(activation.proto, pc - 1, store.access.object, store.access.key, store.value) {
            return Ok(Flow::Done);
        }
        self.set_slow(pc, store)
    }

    fn index_slow(&mut self, activation: Activation, pc: usize, target: u8, access: Access) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let value = self.index(access.object, access.key, access.operand)?;
        Ok(self.store(activation, target, value))
    }

    fn self_slow(&mut self, activation: Activation, pc: usize, a: u8, access: Access) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        self.set_reg(activation, a + 1, access.object);
        let method = self.index(access.object, access.key, access.operand)?;
        Ok(self.store(activation, a, method))
    }

    fn set_slow(&mut self, pc: usize, store: Store) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let access = store.access;
        if self.set_index(access.object, access.key, store.value, access.operand)? {
            Ok(Flow::Next)
        } else {
            Ok(Flow::Reload)
        }
    }

    fn arith_slow(&mut self, activation: Activation, pc: usize, target: u8, arithmetic: Arithmetic) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let value = self.arith_flipped(
            arithmetic.op,
            arithmetic.left,
            arithmetic.right,
            arithmetic.operands,
            arithmetic.flip,
        )?;
        Ok(self.store(activation, target, value))
    }

    fn unary_slow(&mut self, activation: Activation, pc: usize, a: u8, op: UnaryOp, b: u8) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let value = self.reg(activation, b);
        let result = self.unary(op, value, Operand::Register(b))?;
        Ok(self.store(activation, a, result))
    }

    fn equal_slow(&mut self, activation: Activation, pc: &mut usize, left: Value, right: Value, expected: bool) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        let outcome = self.equals(left, right)?;
        Ok(self.compare_jump(activation, pc, outcome, expected))
    }

    fn less_than_slow(&mut self, activation: Activation, pc: &mut usize, left: Value, right: Value, expected: bool) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        let outcome = self.less_than(left, right)?;
        Ok(self.compare_jump(activation, pc, outcome, expected))
    }

    fn less_equal_slow(&mut self, activation: Activation, pc: &mut usize, left: Value, right: Value, expected: bool) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        let outcome = self.less_equal(left, right)?;
        Ok(self.compare_jump(activation, pc, outcome, expected))
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn concat_slow(&mut self, activation: Activation, pc: usize, a: u8, n: u8) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        if self.concat_loop(activation.base, a, usize::from(n))? {
            Ok(Flow::Next)
        } else {
            Ok(Flow::Reload)
        }
    }

    fn close_slow(&mut self, activation: Activation, pc: &mut usize, a: u8) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        let level = activation.base + usize::from(a);
        self.close_upvalues(level);
        if !self.has_to_close(level) {
            return Ok(Flow::Next);
        }
        let caller = self.current_thread;
        let depth = self.frame_count();
        self.close_next(level)?;
        if self.call_settled(caller, depth) {
            *pc -= 1;
            self.save_pc(*pc);
        }
        Ok(Flow::Reload)
    }

    fn err_nil(&mut self, activation: Activation, pc: usize, a: u8, bx: u32) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let slot = activation.base + usize::from(a);
        if self.stack_get(slot).is_nil() {
            return Ok(Flow::Next);
        }
        let frame = self.top_frame().expect("a Lua frame runs ErrNil");
        let proto = self.proto_of(frame.closure);
        let name = match bx.checked_sub(1) {
            Some(index) => super::naming::constant_text(&self.heap, proto, index),
            None => None,
        }
        .unwrap_or_else(|| b"?".to_vec());
        let message = format!("global '{}' already defined", String::from_utf8_lossy(&name));
        Err(self.runtime_error_text(&message))
    }

    fn tbc_slow(&mut self, pc: usize, a: u8) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        self.mark_to_close(a)?;
        Ok(Flow::Next)
    }

    #[inline(always)]
    fn call_closure(&mut self, activation: Activation, pc: usize, closure: Ref<Closure>, a: u8, b: u8, c: u8) -> Result<usize, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        self.record_call(activation.proto, pc - 1, Value::Closure(closure));
        self.push_lua_frame_from(Some(pc), closure, function_slot, argument_count, i32::from(c) - 1, 0)
    }

    #[inline(always)]
    fn enter_called<const CHARGES: bool>(&mut self, window: &Window, pc: usize, closure: Ref<Closure>, a: u8, b: u8, c: u8) -> Result<Window, LuaError> {
        let function_slot = window.base() + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        let nresults = i32::from(c) - 1;
        let base = function_slot + 1;
        if CHARGES {
            self.record_call(window.proto(), pc - 1, Value::Closure(closure));
        }

        let object = Window::closure_object(self, closure);
        let exec = object.code.get();
        let num_params = usize::from(exec.num_params);
        let thread = window.thread();
        if b == 0 || exec.feedback.is_empty() || base > thread.call_limit {
            self.push_lua_frame_from(Some(pc), closure, function_slot, argument_count, nresults, 0)?;
            return Ok(Window::enter(self, window.record()));
        }
        if argument_count < num_params {
            thread.stack[base + argument_count..base + num_params].fill(Value::Nil);
        }
        window.save_pc(pc);
        let frame = LuaFrame::entered(closure, base, LuaFrame::call_status(nresults, 0) | frame_status::NESTED, (object.code, UpvaluesRef::of(object))).with_arguments(argument_count);
        super::window::push_frame(&mut thread.frames, Frame::Lua(frame));
        Ok(window.called(thread))
    }

    /// Follows `luaD_pretailcall` in ldo.c; `None` leaves the call to `tail_call_slow`.
    #[inline(always)]
    fn tail_called(&mut self, window: &Window, closure: Ref<Closure>, a: u8, b: u8, k: bool) -> Option<Window> {
        let function_slot = window.base() + usize::from(a);
        let argument_count = self.value_count(function_slot + 1, b)?;
        if k {
            self.close_upvalues(window.base());
        }
        let object = Window::closure_object(self, closure);
        let exec = object.code.get();
        let thread = window.thread();
        let frame = super::window::top_lua_frame(&mut thread.frames);
        let ret = frame.ret();
        let base = ret + 1;
        let kept = frame.status() & (frame_status::NRESULTS | frame_status::COUNTED | frame_status::NESTED);

        if exec.feedback.is_empty() || base > thread.call_limit {
            return None;
        }
        window.move_down(function_slot, ret, argument_count + 1);
        let frame = super::window::top_lua_frame(&mut thread.frames);
        *frame = LuaFrame::entered(closure, base, kept | frame_status::TAIL, (object.code, UpvaluesRef::of(object))).with_arguments(argument_count);
        let next = window.called(thread);
        next.fill_missing(argument_count, usize::from(exec.num_params));
        Some(next)
    }

    fn call_native_at(&mut self, activation: Activation, pc: usize, native: Ref<crate::object::Native>, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        let caller = self.current_thread;
        let depth = self.frame_count();

        self.save_pc(pc);
        self.record_call(activation.proto, pc - 1, Value::Native(native));
        let request = CallRequest {
            function_slot,
            argument_count,
            nresults: i32::from(c) - 1,
            ccmt: 0,
            operand: Operand::Register(a),
        };
        let quick = !self.tracing && argument_count <= super::ARGUMENTS_INLINE;
        let started = match self.native_body(native) {
            crate::object::NativeBody::Plain(function) if quick => {
                self.call_plain(native, function, (function_slot, argument_count, request.nresults))?;
                CallStart::Done
            }
            crate::object::NativeBody::Stepped(function) if quick => self.call_stepped(native, function, (function_slot, argument_count, request.nresults))?,
            _ => self.start_native(native, request)?,
        };
        match started {
            CallStart::Done if self.call_settled(caller, depth) => return Ok(Flow::Next),
            CallStart::Done => self.call_loop(None)?,
            CallStart::Lua => {}
            CallStart::Callee(next) => self.call_loop(Some(next))?,
        }
        if self.call_settled(caller, depth) {
            return Ok(Flow::Next);
        }
        Ok(Flow::Reload)
    }

    fn call_builtin_at(&mut self, activation: Activation, pc: usize, native: Ref<crate::object::Native>, builtin: Builtin, (a, b, c): (u8, u8, u8)) -> Result<Flow, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        let wanted = i32::from(c) - 1;
        let switched = match builtin {
            Builtin::Resume | Builtin::Wrap => {
                self.resume_fast(native, builtin == Builtin::Wrap, function_slot, argument_count, wanted, pc)
            }
            Builtin::Yield => {
                self.yield_fast(native, function_slot, argument_count, wanted, pc)
            }
            _ if self.call_builtin(builtin, function_slot, argument_count, wanted) => return Ok(Flow::Next),
            _ => false,
        };
        if !switched {
            return self.call_native_at(activation, pc, native, a, b, c);
        }
        if !self.switch_landed_idle() {
            self.call_loop(None)?;
        }
        Ok(Flow::Reload)
    }

    /// Follows LuaJIT's `coroutine_resume_wrap` in `vm_x64.dasc`; a stackless VM continues in the same loop instead of nesting the call.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn switch_at(&mut self, activation: Activation, pc: usize, native: Ref<crate::object::Native>, builtin: Builtin, (a, b, c): (u8, u8, u8), scope: &RunScope) -> Result<Switched, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        let wanted = i32::from(c) - 1;

        let switched = match builtin {
            Builtin::Yield => self.yield_fast(native, function_slot, argument_count, wanted, pc),
            _ => self.resume_fast(native, builtin == Builtin::Wrap, function_slot, argument_count, wanted, pc),
        };
        if !switched {
            return Ok(Switched::Not);
        }
        match self.landing(scope) {
            Some(switched) => Ok(switched),
            None => {
                self.call_loop(None)?;
                Ok(Switched::Left)
            }
        }
    }

    #[inline(always)]
    fn landing(&self, scope: &RunScope) -> Option<Switched> {
        let current = self.current_thread;
        let frames = &self.heap.threads.live(current).frames;
        let Some(Frame::Lua(frame)) = frames.last() else {
            return None;
        };
        if !frame.is_idle() {
            return None;
        }
        let (closure, base, pc) = (frame.closure, frame.base, frame.pc());
        if (scope.return_on_switch && current != scope.thread) || (current == scope.thread && frames.len() <= scope.frames) {
            return Some(Switched::Left);
        }
        Some(Switched::Landed(Entry { closure, base, pc }))
    }

    /// False leaves the return to `return_values`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn return_to_idle_caller(&mut self, first: usize, count: usize) -> bool {
        let thread = self.thread_mut();
        let [.., Frame::Lua(caller), Frame::Lua(callee)] = thread.frames.as_slice() else {
            return false;
        };
        if callee.counted() || !caller.is_idle() {
            return false;
        }
        let (destination, wanted) = (callee.ret(), callee.nresults());
        super::move_results(&mut thread.stack, first, count, destination, wanted);
        if wanted < 0 {
            thread.top = destination + count;
        }
        super::window::pop_lua_frame(&mut thread.frames);
        true
    }

    /// Follows `returning` in `luaV_execute`; a nested frame's caller ran in the same run, so the return never ends its scope.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn return_into_caller(&mut self, window: &Window, first: usize, count: usize, scope: &RunScope) -> Option<(Window, usize)> {
        let (nested, destination, wanted) = window.returns_to();
        if !nested {
            return None;
        }
        let thread = window.thread();

        window.move_results(first, count, destination, wanted);
        if wanted < 0 {
            thread.top = destination + count;
        }
        super::window::pop_lua_frame(&mut thread.frames);
        debug_assert!(thread.frames.len() > scope.frames || self.current_thread != scope.thread, "a nested frame's return ends no run");
        let caller = super::window::top_lua_frame(&mut thread.frames);
        debug_assert!(caller.is_idle() && !caller.code.is_null());
        let pc = caller.pc();
        Some((window.returned(thread), pc))
    }

    #[inline(always)]
    fn top_entry(&self, scope: &RunScope, step: bool, hooks: bool) -> TopEntry {
        if scope.return_on_switch && self.current_thread != scope.thread {
            return TopEntry::Leave(false);
        }
        let frames = &self.thread().frames;
        if self.current_thread == scope.thread && frames.len() <= scope.frames {
            return TopEntry::Leave(false);
        }
        if !step && self.tracing != hooks {
            return TopEntry::Leave(true);
        }
        match frames.last() {
            Some(Frame::Lua(frame)) => TopEntry::Lua(Entry {
                closure: frame.closure,
                base: frame.base,
                pc: frame.pc(),
            }),
            _ => TopEntry::Coroutine,
        }
    }

    fn call_slow(&mut self, activation: Activation, pc: usize, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        self.call_instruction(activation, pc - 1, a, b, c)
    }

    /// `needclose` is the `k` flag: the frame has upvalues to close.
    fn tail_call_slow(&mut self, activation: Activation, pc: usize, a: u8, b: u8, needclose: bool) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        self.tail_call(activation, a, b, needclose)
    }

    fn return_slow(&mut self, activation: Activation, pc: &mut usize, a: u8, b: u8, needclose: bool) -> Result<Flow, LuaError> {
        let first = activation.base + usize::from(a);
        let count = self.checked_value_count(first, b)?;
        self.return_values(activation, first, count, needclose, pc)
    }

    fn for_prep_slow(&mut self, activation: Activation, pc: &mut usize, a: u8, jump: u32) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        if self.for_prepare(activation, a)? {
            *pc += jump as usize + 1;
        }
        Ok(Flow::Next)
    }

    fn tfor_prep_slow(&mut self, activation: Activation, pc: &mut usize, a: u8, jump: u32) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        let control_slot = activation.base + usize::from(a) + 3;
        let closing_slot = control_slot - 1;
        let control = self.stack_get(control_slot);
        let closing = self.stack_get(closing_slot);
        self.stack_set(closing_slot, control);
        self.stack_set(control_slot, closing);
        self.mark_to_close(a + 2)?;
        *pc += jump as usize;
        Ok(Flow::Untraced)
    }

    /// A pending to-be-closed variable needs the generic return for its `__close` call.
    #[inline(never)]
    pub(super) fn closed_for_return(&mut self, base: usize) -> bool {
        if self.has_to_close(base) {
            return false;
        }
        self.close_upvalues(base);
        true
    }

    /// A `top` below the first value only comes from bytecode that uses `top` without setting it.
    #[inline(always)]
    fn value_count(&self, first: usize, b: u8) -> Option<usize> {
        if b != 0 {
            return Some(usize::from(b) - 1);
        }
        self.thread().top.checked_sub(first)
    }

    #[inline(always)]
    fn checked_value_count(&mut self, first: usize, b: u8) -> Result<usize, LuaError> {
        match self.value_count(first, b) {
            Some(count) => Ok(count),
            None => Err(self.runtime_error(b"invalid instruction")),
        }
    }

    fn call_instruction(&mut self, activation: Activation, site: usize, a: u8, b: u8, c: u8) -> Result<Flow, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        let nresults = i32::from(c) - 1;
        let callee = self.stack_get(function_slot);
        self.record_call(activation.proto, site, callee);
        if let Value::Closure(closure) = callee
            && !self.tracing
        {
            self.push_lua_frame(closure, function_slot, argument_count, nresults, 0)?;
            return Ok(Flow::Reload);
        }
        let caller = self.current_thread;
        let depth = self.frame_count();
        self.call_loop(Some(CallRequest {
            function_slot,
            argument_count,
            nresults,
            ccmt: 0,
            operand: Operand::Register(a),
        }))?;
        if self.call_settled(caller, depth) {
            return Ok(Flow::Next);
        }
        Ok(Flow::Reload)
    }

    fn return_values(
        &mut self,
        activation: Activation,
        first: usize,
        count: usize,
        needclose: bool,
        pc: &mut usize,
    ) -> Result<Flow, LuaError> {
        self.save_pc(*pc);
        if needclose && self.has_to_close(activation.base) {
            self.thread_mut().top = first + count;
            let caller = self.current_thread;
            let depth = self.frame_count();
            self.close_next(activation.base)?;
            if self.call_settled(caller, depth) {
                *pc -= 1;
                self.save_pc(*pc);
            }
            return Ok(Flow::Reload);
        }
        self.finish_return(first, count, needclose)?;
        if !self.caller_resumes_idle() {
            self.call_loop(None)?;
        }
        Ok(Flow::Reload)
    }

    fn caller_resumes_idle(&self) -> bool {
        matches!(self.thread().frames.last(), Some(Frame::Lua(caller)) if caller.is_idle())
    }

    fn tail_call(&mut self, activation: Activation, a: u8, b: u8, needclose: bool) -> Result<Flow, LuaError> {
        let function_slot = activation.base + usize::from(a);
        let argument_count = self.checked_value_count(function_slot + 1, b)?;
        if let Some((levels, closure)) = self.lua_callee_behind_call_metamethods(self.stack_get(function_slot)) {
            let argument_count = self.insert_call_metamethods(function_slot, argument_count, levels);
            self.tail_into(activation, closure, function_slot, argument_count, needclose)?;
            return Ok(Flow::Reload);
        }
        let caller = self.current_thread;
        let depth = self.frame_count();
        if !self.tail_builtin(function_slot, argument_count) {
            self.call_loop(Some(CallRequest {
                function_slot,
                argument_count,
                nresults: -1,
                ccmt: 0,
                operand: Operand::Register(a),
            }))?;
        }
        if self.call_settled(caller, depth) {
            let top = self.thread().top;
            self.finish_return(function_slot, top - function_slot, needclose)?;
            self.call_loop(None)?;
        } else {
            self.set_pending_in(caller, depth - 1, Pending::Return(function_slot));
        }
        Ok(Flow::Reload)
    }

    fn tail_builtin(&mut self, function_slot: usize, argument_count: usize) -> bool {
        let Value::Native(native) = self.stack_get(function_slot) else {
            return false;
        };
        if self.tracing {
            return false;
        }
        let Some(builtin) = self.builtin_of(native) else {
            return false;
        };
        self.call_builtin(builtin, function_slot, argument_count, -1)
    }

    /// `None` leaves the call to the ordinary path, which also raises the errors for a value that cannot be called.
    fn lua_callee_behind_call_metamethods(&mut self, callee: Value) -> Option<(u8, Ref<Closure>)> {
        let mut levels = 0;
        let mut object = callee;
        loop {
            if let Value::Closure(closure) = object {
                return Some((levels, closure));
            }
            if levels == MAX_CALL_METAMETHODS {
                return None;
            }
            object = self.metamethod(object, Event::Call);
            if object.is_nil() {
                return None;
            }
            levels += 1;
        }
    }

    /// Applies `tryfuncTM` `levels` times, leaving the closure in `function_slot`.
    fn insert_call_metamethods(&mut self, function_slot: usize, mut argument_count: usize, levels: u8) -> usize {
        for _ in 0..levels {
            let handler = self.metamethod(self.stack_get(function_slot), Event::Call);
            self.ensure_stack(function_slot + argument_count + 2);
            for offset in (0..=argument_count).rev() {
                let value = self.stack_get(function_slot + offset);
                self.stack_set(function_slot + offset + 1, value);
            }
            self.stack_set(function_slot, handler);
            argument_count += 1;
        }
        argument_count
    }

    fn tail_into(&mut self, activation: Activation, closure: Ref<Closure>, function_slot: usize, argument_count: usize, needclose: bool) -> Result<(), LuaError> {
        let frame = *self.top_frame().expect("a Lua frame makes the tail call");
        let (ret, nresults, counted, nested) = (frame.ret(), frame.nresults(), frame.counted(), frame.nested());
        if needclose {
            self.close_upvalues(activation.base);
        }
        for offset in 0..=argument_count {
            let value = self.stack_get(function_slot + offset);
            self.stack_set(ret + offset, value);
        }
        self.pop_top_frame();
        self.push_lua_frame(closure, ret, argument_count, nresults, 0)?;
        if let Some(Frame::Lua(called)) = self.thread_mut().frames.last_mut() {
            let status = called.status() & !frame_status::NESTED;
            called.set_status(status | frame_status::TAIL | if counted { frame_status::COUNTED } else { 0 } | if nested { frame_status::NESTED } else { 0 });
        }
        self.hook_call(true)
    }

    fn generic_call(&mut self, activation: Activation, pc: usize, a: u8, c: u8) -> Result<Flow, LuaError> {
        self.save_pc(pc);
        let base = activation.base + usize::from(a);
        let control = self.stack_get(base + 3);
        let state = self.stack_get(base + 1);
        let function = self.stack_get(base);
        self.stack_set(base + 5, control);
        self.stack_set(base + 4, state);
        self.stack_set(base + 3, function);
        let function_slot = base + 3;
        self.thread_mut().top = function_slot + 3;
        if let Value::Native(native) = function
            && !self.tracing
            && let Some(builtin) = self.builtin_of(native)
            && self.generic_builtin(builtin, base, c)
        {
            return Ok(Flow::Untraced);
        }
        let caller = self.current_thread;
        let depth = self.frame_count();
        let request = CallRequest {
            function_slot,
            argument_count: 2,
            nresults: i32::from(c),
            ccmt: 0,
            operand: Operand::Unknown,
        };
        let body = match function {
            Value::Native(native) if !self.tracing => Some((native, self.native_body(native))),
            _ => None,
        };
        let started = match body {
            Some((native, crate::object::NativeBody::Plain(function))) => {
                self.call_plain(native, function, (function_slot, 2, request.nresults))?;
                Some(CallStart::Done)
            }
            Some((native, crate::object::NativeBody::Stepped(function))) => Some(self.call_stepped(native, function, (function_slot, 2, request.nresults))?),
            _ => None,
        };
        match started {
            Some(CallStart::Done) if self.call_settled(caller, depth) => return Ok(Flow::Untraced),
            Some(CallStart::Done) => self.call_loop(None)?,
            Some(CallStart::Lua) => {}
            Some(CallStart::Callee(next)) => self.call_loop(Some(next))?,
            None => self.call_loop(Some(request))?,
        }
        if self.call_settled(caller, depth) {
            return Ok(Flow::Untraced);
        }
        Ok(Flow::Reload)
    }

    fn set_list(&mut self, activation: Activation, a: u8, b: u8, stored: u32) -> Result<(), LuaError> {
        let first = activation.base + usize::from(a);
        let count = if b == 0 {
            self.checked_value_count(first + 1, 0)?
        } else {
            usize::from(b)
        };
        if count == 0 {
            return Ok(());
        }
        if let Value::Table(table) = self.reg(activation, a) {
            let thread = self.current_thread;
            let slots = first + 1..first + 1 + count;
            let first_key = stored as usize + 1;
            let mut outcome = self.heap.table_set_array_from_thread(table, first_key, thread, slots.clone());
            if outcome == Err(SetError::Memory) {
                self.gc_collect();
                outcome = self.heap.table_set_array_from_thread(table, first_key, thread, slots);
            }
            if outcome.is_err() {
                return Err(self.memory_error());
            }
        }
        Ok(())
    }

    pub(super) fn new_closure(&mut self, window: Passed<'_>, index: usize) -> Ref<Closure> {
        self.new_closure_in(Activation { closure: window.closure(), base: window.base(), proto: window.proto() }, index)
    }

    fn new_closure_in(&mut self, activation: Activation, index: usize) -> Ref<Closure> {
        let child = self.heap.protos.live(activation.proto).protos[index];
        let function = self.heap.protos.live(child);
        let count = function.upvalues.len();
        let code = super::CodeRef::of(&function.exec);

        let mut upvalues = Vec::with_capacity(count);
        for position in 0..count {
            let captured = self.capture(activation.closure, activation.base, child, position);
            upvalues.push(captured);
        }
        self.heap.alloc(Closure { proto: child, upvalues, code })
    }

    /// Follows `pushclosure` in lvm.c.
    fn capture(&mut self, parent_closure: Ref<Closure>, base: usize, child: Ref<Proto>, position: usize) -> Captured {
        let description = &self.heap.protos.live(child).upvalues[position];
        let (in_stack, slot) = (description.in_stack, usize::from(description.index));

        if !in_stack {
            return self.heap.closures.live(parent_closure).upvalues[slot];
        }
        let upvalue = self.find_or_open_upvalue(base + slot);
        self.heap.captured(upvalue)
    }

    /// Follows `luaT_adjustvarargs` in ltm.c; a vararg table or a stack that must grow takes `adjust_varargs`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn prepare_varargs(&mut self, window: &Window) -> Result<Window, LuaError> {
        let exec = window.exec();
        let (params, vararg_table) = (exec.num_params, exec.vararg_table);
        let thread = window.thread();
        let frame = super::window::top_lua_frame(&mut thread.frames);
        let base = frame.base;
        let total = frame.argument_count().max(usize::from(params));
        let moved = base + total + 1;

        if vararg_table || moved > thread.call_limit {
            self.adjust_varargs(Activation::of(window))?;
            return Ok(Window::enter(self, window.record()));
        }
        hide_arguments(&mut thread.stack, base, moved, usize::from(params));
        frame.base = moved;
        frame.set_hidden(params, total - usize::from(params));
        Ok(window.called(thread))
    }

    /// Follows `luaT_getvarargs` in ltm.c. False leaves the copy to `copy_varargs` when the values reach past the window.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn copy_hidden_varargs(window: &Window, a: u8, c: u8) -> bool {
        let thread = window.thread();
        let available = super::window::top_lua_frame(&mut thread.frames).nvarargs();
        let wanted = if c == 0 { available } else { usize::from(c) - 1 };
        let base = window.base();
        let destination = base + usize::from(a);
        let source = base - 1 - available;

        if usize::from(a) + wanted > super::REGISTER_SPAN {
            return false;
        }
        let copied = wanted.min(available);
        let stack = &mut thread.stack[source..destination + wanted];
        let offset = destination - source;
        for index in 0..copied {
            stack[offset + index] = stack[index];
        }
        stack[offset + copied..].fill(Value::Nil);
        if c == 0 {
            thread.top = destination + available;
        }
        true
    }

    /// Follows `luaT_adjustvarargs` in ltm.c.
    fn adjust_varargs(&mut self, activation: Activation) -> Result<(), LuaError> {
        let passed = self.top_frame().expect("a Lua frame runs VARARGPREP").argument_count();
        let function = self.heap.protos.live(activation.proto);
        let (num_params, vararg_table, max_stack) = (function.num_params, function.vararg_table, usize::from(function.max_stack));
        let params = usize::from(num_params);
        let total = passed.max(params);
        let extra = total - params;
        let base = activation.base;

        if vararg_table {
            return self.build_vararg_table(base + params, extra);
        }
        let moved = base + total + 1;
        self.ensure_stack(moved + super::REGISTER_SPAN.max(max_stack));
        let thread = self.thread_mut();
        hide_arguments(&mut thread.stack, base, moved, params);
        let frame = super::window::top_lua_frame(&mut thread.frames);
        frame.base = moved;
        frame.set_hidden(num_params, extra);
        Ok(())
    }

    /// Follows PUC's `createvarargtab`.
    fn build_vararg_table(&mut self, slot: usize, extra: usize) -> Result<(), LuaError> {
        let table = self.new_table_with_sizes(extra, 1)?;
        for index in 0..extra {
            let value = self.stack_get(slot + index);
            self.raw_set(table, Value::Integer(index as i64 + 1), value)?;
        }
        let count_key = Value::String(self.heap.string(b"n"));
        self.raw_set(table, count_key, Value::Integer(extra as i64))?;
        self.stack_set(slot, Value::Table(table));
        Ok(())
    }

    fn get_vararg(&mut self, activation: Activation, a: u8, c: u8) {
        let key = self.reg(activation, c);
        let available = self.top_frame().expect("a Lua frame runs VARARG").nvarargs();
        let index = match key {
            Value::Integer(index) => Some(index),
            Value::Float(number) => number::float_to_integer(number, Rounding::Exact),
            _ => None,
        };
        let value = match index {
            Some(index) if index >= 1 && (index as u64) <= available as u64 => {
                self.stack_get(activation.base - 1 - available + index as usize - 1)
            }
            Some(_) => Value::Nil,
            None if self.string_bytes(key) == Some(&b"n"[..]) => Value::Integer(available as i64),
            None => Value::Nil,
        };
        self.set_reg(activation, a, value);
    }

    fn copy_varargs(&mut self, activation: Activation, a: u8, c: u8, table: Option<u8>) -> Result<(), LuaError> {
        let destination = activation.base + usize::from(a);
        let table = table.map(|register| self.vararg_table(activation, register)).transpose()?;
        let available = match table {
            Some(table) => self.vararg_table_count(table)?,
            None => self.top_frame().expect("a Lua frame runs VARARG").nvarargs(),
        };
        let wanted = if c == 0 { available } else { usize::from(c) - 1 };
        self.ensure_stack(destination + wanted);
        for index in 0..wanted {
            let value = self.vararg_value(activation, table, available, index);
            self.stack_set(destination + index, value);
        }
        if c == 0 {
            self.thread_mut().top = destination + available;
        }
        Ok(())
    }

    fn vararg_table(&mut self, activation: Activation, register: u8) -> Result<Ref<Table>, LuaError> {
        match self.reg(activation, register) {
            Value::Table(table) => Ok(table),
            _ => Err(self.runtime_error(b"invalid instruction")),
        }
    }

    fn vararg_table_count(&mut self, table: Ref<Table>) -> Result<usize, LuaError> {
        let key = Value::String(self.heap.string(b"n"));
        match self.table_get(table, key) {
            Value::Integer(count) if count >= 0 && (count as u64) <= (i32::MAX / 2) as u64 => Ok(count as usize),
            _ => Err(self.runtime_error(b"vararg table has no proper 'n'")),
        }
    }

    fn vararg_value(&self, activation: Activation, table: Option<Ref<Table>>, available: usize, index: usize) -> Value {
        if index >= available {
            return Value::Nil;
        }
        match table {
            Some(table) => self.table_get(table, Value::Integer(index as i64 + 1)),
            None => self.stack_get(activation.base - 1 - available + index),
        }
    }

    fn for_prepare(&mut self, activation: Activation, a: u8) -> Result<bool, LuaError> {
        let initial = self.reg(activation, a);
        let limit = self.reg(activation, a + 1);
        let step = self.reg(activation, a + 2);
        if let (Value::Integer(start), Value::Integer(increment)) = (initial, step) {
            if increment == 0 {
                return Err(self.runtime_error(b"'for' step is zero"));
            }
            let Some(last) = self.for_limit(start, limit, increment)? else {
                return Ok(true);
            };
            let count = int_loop_count(start, last, increment);
            self.set_reg(activation, a, Value::Integer(count as i64));
            self.set_reg(activation, a + 1, Value::Integer(increment));
            self.set_reg(activation, a + 2, Value::Integer(start));
            return Ok(false);
        }
        let float_limit = self.for_number(limit, "limit")?;
        let float_step = self.for_number(step, "step")?;
        let float_start = self.for_number(initial, "initial value")?;
        if float_step == 0.0 {
            return Err(self.runtime_error(b"'for' step is zero"));
        }
        let runs = if 0.0 >= float_step {
            float_limit <= float_start
        } else {
            float_start <= float_limit
        };
        if !runs {
            return Ok(true);
        }
        self.set_reg(activation, a, Value::Float(float_limit));
        self.set_reg(activation, a + 1, Value::Float(float_step));
        self.set_reg(activation, a + 2, Value::Float(float_start));
        Ok(false)
    }

    fn for_number(&mut self, value: Value, what: &str) -> Result<f64, LuaError> {
        match self.coerce_for_number(value) {
            Some(number) => Ok(number.to_float()),
            None => {
                let type_name = self.object_type_name(value);
                let message = format!("bad 'for' {what} (number expected, got {type_name})");
                Err(self.runtime_error(message.as_bytes()))
            }
        }
    }

    fn coerce_for_number(&self, value: Value) -> Option<Number> {
        match value {
            Value::Integer(integer) => Some(Number::Integer(integer)),
            Value::Float(float) => Some(Number::Float(float)),
            Value::String(string) => number::parse_number(self.heap.bytes(string)),
            _ => None,
        }
    }

    fn for_limit(&mut self, start: i64, limit: Value, step: i64) -> Result<Option<i64>, LuaError> {
        let last = match self.coerce_for_number(limit) {
            Some(Number::Integer(integer)) => integer,
            Some(Number::Float(float)) => {
                let rounding = if step < 0 { Rounding::Ceil } else { Rounding::Floor };
                match number::float_to_integer(float, rounding) {
                    Some(integer) => integer,
                    None if float.is_nan() => return Ok(None),
                    None if float > 0.0 => {
                        if step < 0 {
                            return Ok(None);
                        }
                        i64::MAX
                    }
                    None => {
                        if step > 0 {
                            return Ok(None);
                        }
                        i64::MIN
                    }
                }
            }
            None => {
                let type_name = self.object_type_name(limit);
                let message = format!("bad 'for' limit (number expected, got {type_name})");
                return Err(self.runtime_error(message.as_bytes()));
            }
        };
        let skip = if step > 0 { start > last } else { start < last };
        Ok((!skip).then_some(last))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    #[test]
    fn a_tail_call_through_call_metamethods_reuses_the_frame() {
        let mut vm = Vm::new(Options::default());
        vm.run_source(
            b"=t",
            b"local mt = {}\n\
              local obj = setmetatable({}, mt)\n\
              mt.__call = function(self, k) if k == 0 then return 'done' end return obj(k - 1) end\n\
              assert(obj(100000) == 'done')",
        )
        .expect("the tail calls through __call run in constant stack");
    }
}

#[cfg(test)]
mod memory_tests {
    use crate::{Options, Vm};

    fn refused(source: &[u8]) -> bool {
        let mut vm = Vm::new(Options {
            memory_limit: Some(64 << 20),
            ..Options::default()
        });
        match vm.run_source(b"=t", source) {
            Ok(()) => false,
            Err(error) => vm.error_message(&error).windows(b"not enough memory".len()).any(|window| window == b"not enough memory"),
        }
    }

    #[test]
    fn one_large_allocation_is_refused_before_it_is_made() {
        assert!(refused(b"table.create(1 << 27)"));
        assert!(refused(b"string.rep('x', 1 << 31)"));
        assert!(refused(b"local t = {} for i = 1, 3 do t[i] = i end local s = 'x' .. string.rep('y', 1 << 25) .. 'z' .. string.rep('w', 1 << 25)"));
    }

    fn memory_error_text(vm: &mut Vm, source: &[u8]) -> bool {
        match vm.run_source(b"=t", source) {
            Ok(()) => false,
            Err(error) => vm.error_message(&error).windows(b"not enough memory".len()).any(|window| window == b"not enough memory"),
        }
    }

    #[test]
    fn an_allocation_the_allocator_refuses_is_a_memory_error_without_a_limit() {
        let mut vm = Vm::new(Options::default());
        assert!(crate::table::Table::with_sizes(1 << 45, 0).is_err());
        assert!(memory_error_text(&mut vm, b"string.rep('x', 1 << 40)"));
    }

    #[test]
    fn table_growth_is_held_to_the_memory_limit() {
        let mut vm = Vm::new(Options {
            memory_limit: Some(16 << 20),
            ..Options::default()
        });
        assert!(memory_error_text(&mut vm, b"local t = {} for i = 1, 10000000 do t[i] = i end"));
        assert!(vm.heap.gc.total <= 16 << 20);
    }

    #[test]
    fn a_newtable_size_beyond_the_array_limit_is_a_memory_error() {
        let mut vm = Vm::new(Options::default());
        assert!(vm.new_table_sized(1 << 32, 0, false).is_err());
        assert!(vm.new_table_sized(0, 32, false).is_err());
    }
}

/// Follows `buildhiddenargs` in ltm.c. The two ranges do not overlap, since `moved` is past every argument.
fn hide_arguments(stack: &mut [Value], base: usize, moved: usize, params: usize) {
    let stack = &mut stack[base - 1..=moved + params];
    let offset = moved - base;
    for index in 0..=params {
        stack[offset + index] = stack[index];
    }
    stack[1..=params].fill(Value::Nil);
    stack[offset + params + 1] = Value::Nil;
}
