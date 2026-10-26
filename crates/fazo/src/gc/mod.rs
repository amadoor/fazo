mod mark;
mod param;
mod sweep;

pub(crate) use param::{MAX_LMEM, PARAM_COUNT, Param, apply_param, code_param};

use crate::vm::Event;
use std::collections::VecDeque;

use crate::{
    LuaError, Vm, values,
    arena::Ref,
    object::Native,
    table::Table,
    value::Value,
};

pub(crate) const AGE_MASK: u8 = 0b111;
pub(crate) const WHITE0: u8 = 1 << 3;
pub(crate) const WHITE1: u8 = 1 << 4;
pub(crate) const BLACK: u8 = 1 << 5;
pub(crate) const WHITE_BITS: u8 = WHITE0 | WHITE1;
pub(crate) const COLOR_BITS: u8 = BLACK | WHITE_BITS;
pub(crate) const GC_BITS: u8 = COLOR_BITS | AGE_MASK;
pub(crate) const FINALIZED: u8 = 1 << 6;

pub(crate) const NEW: u8 = 0;
pub(crate) const SURVIVAL: u8 = 1;
pub(crate) const OLD0: u8 = 2;
pub(crate) const OLD1: u8 = 3;
pub(crate) const OLD: u8 = 4;
pub(crate) const TOUCHED1: u8 = 5;
pub(crate) const TOUCHED2: u8 = 6;

pub(crate) const SWEEP_MAX: usize = 20;
const CWUFIN: i64 = 10;
const STOPPED_DEBT: i64 = 20000;
const POINTER_SIZE: i64 = 8;

pub(crate) const STOP_USER: u8 = 1;
pub(crate) const STOP_COLLECTOR: u8 = 2;
pub(crate) const STOP_CLOSING: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Gc {
    String(Ref<crate::object::LuaString>),
    Table(Ref<Table>),
    Closure(Ref<crate::object::Closure>),
    Native(Ref<Native>),
    UpVal(Ref<crate::object::UpVal>),
    Proto(Ref<crate::proto::Proto>),
    Thread(Ref<crate::object::Thread>),
    Userdata(Ref<crate::object::Userdata>),
    Shape(Ref<crate::shape::Shape>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub(crate) enum Phase {
    Pause,
    Propagate,
    EnterAtomic,
    Atomic,
    SweepAllgc,
    SweepFinobj,
    SweepTobefnz,
    SweepEnd,
    CallFinalizers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Kind {
    Inc,
    GenMinor,
    GenMajor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepResult {
    Work(i64),
    Step2Minor,
    AtomicStep,
    Step2Pause,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct GcState {
    pub(crate) total: i64,
    pub(crate) limit: Option<usize>,
    pub(crate) debt: i64,
    pub(crate) current_white: u8,
    pub(crate) phase: Phase,
    pub(crate) kind: Kind,
    pub(crate) marked: i64,
    pub(crate) major_minor: i64,
    pub(crate) params: [u8; PARAM_COUNT],
    pub(crate) stop: u8,
    pub(crate) gray: Vec<Gc>,
    pub(crate) grayagain: Vec<Gc>,
    pub(crate) weak: Vec<Gc>,
    pub(crate) allweak: Vec<Gc>,
    pub(crate) ephemeron: Vec<Gc>,
    pub(crate) young: Vec<Gc>,
    #[serde(skip)]
    pub(crate) spare_young: Vec<Gc>,
    #[serde(skip)]
    pub(crate) spare_gray: Vec<Gc>,
    #[serde(skip)]
    pub(crate) roots: Vec<Value>,
    pub(crate) old1: Vec<Gc>,
    pub(crate) finobj: Vec<Gc>,
    pub(crate) tobefnz: VecDeque<Gc>,
    pub(crate) sweep_arena: usize,
    pub(crate) sweep_index: usize,
    pub(crate) live_native: Option<(Ref<crate::object::Thread>, usize, usize)>,
    pub(crate) safe_window: Option<(Ref<crate::object::Thread>, usize, usize)>,
    /// Compiled code that allocates leaves when this changes, since a finalizer may change anything.
    pub(crate) finalizers_called: u64,
}

impl Default for GcState {
    fn default() -> Self {
        Self {
            total: 0,
            limit: None,
            debt: 0,
            current_white: WHITE0,
            phase: Phase::Pause,
            kind: Kind::Inc,
            marked: 0,
            major_minor: 0,
            params: Param::ALL.map(|param| code_param(param.default_percent())),
            stop: 0,
            gray: Vec::new(),
            grayagain: Vec::new(),
            weak: Vec::new(),
            allweak: Vec::new(),
            ephemeron: Vec::new(),
            young: Vec::new(),
            spare_young: Vec::new(),
            spare_gray: Vec::new(),
            roots: Vec::new(),
            old1: Vec::new(),
            finobj: Vec::new(),
            tobefnz: VecDeque::new(),
            sweep_arena: 0,
            sweep_index: 0,
            live_native: None,
            safe_window: None,
            finalizers_called: 0,
        }
    }
}

impl GcState {
    pub(crate) fn grow(&mut self, delta: i64) {
        self.total += delta;
        self.debt = pinned_debt(self.debt - delta);
    }

    pub(crate) fn is_generational(&self) -> bool {
        self.kind == Kind::GenMinor
    }

    pub(crate) fn is_generational_mode(&self) -> bool {
        self.kind != Kind::Inc
    }

    pub(crate) fn keeps_invariant(&self) -> bool {
        self.phase <= Phase::Atomic
    }

    pub(crate) fn in_sweep(&self) -> bool {
        matches!(
            self.phase,
            Phase::SweepAllgc | Phase::SweepFinobj | Phase::SweepTobefnz | Phase::SweepEnd
        )
    }

    pub(crate) fn other_white(&self) -> u8 {
        self.current_white ^ WHITE_BITS
    }

    pub(crate) fn running(&self) -> bool {
        self.stop == 0
    }

    pub(crate) fn param(&self, param: Param, x: i64) -> i64 {
        apply_param(self.params[param.index()], x)
    }

    fn clear_gray_lists(&mut self) {
        self.gray.clear();
        self.grayagain.clear();
        self.weak.clear();
        self.allweak.clear();
        self.ephemeron.clear();
    }
}

fn pinned_debt(debt: i64) -> i64 {
    if cfg!(feature = "gc-stress") { debt.min(0) } else { debt }
}

pub(crate) fn is_white(marks: u8) -> bool {
    marks & WHITE_BITS != 0
}

pub(crate) fn is_black(marks: u8) -> bool {
    marks & BLACK != 0
}

pub(crate) fn age(marks: u8) -> u8 {
    marks & AGE_MASK
}

pub(crate) fn is_old(marks: u8) -> bool {
    age(marks) > SURVIVAL
}

pub(crate) fn is_finalized(marks: u8) -> bool {
    marks & FINALIZED != 0
}

pub(crate) fn with_age(marks: u8, age: u8) -> u8 {
    (marks & !AGE_MASK) | age
}

pub(crate) fn make_white(marks: u8, white: u8) -> u8 {
    (marks & !COLOR_BITS) | white
}

pub(crate) fn set_gray(marks: u8) -> u8 {
    marks & !COLOR_BITS
}

pub(crate) fn set_black(marks: u8) -> u8 {
    (marks & !WHITE_BITS) | BLACK
}

pub(crate) fn value_gc(value: Value) -> Option<Gc> {
    match value {
        Value::String(reference) => Some(Gc::String(reference)),
        Value::Table(reference) => Some(Gc::Table(reference)),
        Value::Closure(reference) => Some(Gc::Closure(reference)),
        Value::Native(reference) => Some(Gc::Native(reference)),
        Value::Thread(reference) => Some(Gc::Thread(reference)),
        Value::Userdata(reference) => Some(Gc::Userdata(reference)),
        Value::Nil | Value::Boolean(_) | Value::Integer(_) | Value::Float(_) | Value::LightUserdata(_) => None,
    }
}

pub(crate) fn gc_value(gc: Gc) -> Option<Value> {
    Some(match gc {
        Gc::String(reference) => Value::String(reference),
        Gc::Table(reference) => Value::Table(reference),
        Gc::Closure(reference) => Value::Closure(reference),
        Gc::Native(reference) => Value::Native(reference),
        Gc::Thread(reference) => Value::Thread(reference),
        Gc::Userdata(reference) => Value::Userdata(reference),
        Gc::UpVal(_) | Gc::Proto(_) | Gc::Shape(_) => return None,
    })
}

impl Vm {
    pub(crate) fn gc_safe_point_at(&mut self, base: usize, top: usize) {
        let previous = self.heap.gc.safe_window.replace((self.current_thread, base, top));
        self.gc_step();
        self.heap.gc.safe_window = previous;
    }

    fn note_running_native(&mut self) {
        self.heap.gc.live_native = self
            .current_native
            .map(|call| (self.current_thread, call.slot, call.slot + 1 + call.count));
    }

    fn set_debt(&mut self, debt: i64) {
        self.heap.gc.debt = pinned_debt(debt.min(MAX_LMEM - self.heap.gc.total));
    }

    fn set_pause(&mut self) {
        let threshold = self.heap.gc.param(Param::Pause, self.heap.gc.marked);
        let debt = (threshold - self.heap.gc.total).max(0);
        self.set_debt(debt);
    }

    fn set_minor_debt(&mut self) {
        let debt = self.heap.gc.param(Param::MinorMul, self.heap.gc.major_minor);
        self.set_debt(debt);
    }

    fn gc_step(&mut self) {
        if !self.heap.gc.running() {
            if self.heap.gc.stop & STOP_USER != 0 {
                self.set_debt(STOPPED_DEBT);
            }
            return;
        }
        if cfg!(feature = "gc-stress") {
            self.gc_collect();
            return;
        }
        match self.heap.gc.kind {
            Kind::Inc | Kind::GenMajor => self.inc_step(),
            Kind::GenMinor => {
                self.young_collection();
                self.set_minor_debt();
            }
        }
    }

    fn inc_step(&mut self) {
        let stepsize = self.heap.gc.param(Param::StepSize, 100);
        let mut work2do = self.heap.gc.param(Param::StepMul, stepsize / POINTER_SIZE);
        let fast = work2do == 0;
        loop {
            if !fast && self.heap.gc.phase == Phase::Propagate && !self.heap.gc.gray.is_empty() {
                self.note_running_native();
                work2do -= self.heap.propagate_until(work2do.max(1));
                if work2do <= 0 {
                    break;
                }
                continue;
            }
            match self.single_step(fast) {
                StepResult::Step2Minor => return,
                StepResult::Step2Pause => break,
                StepResult::AtomicStep if !fast => break,
                StepResult::AtomicStep => {}
                StepResult::Work(work) => work2do -= work,
            }
            if !(fast || work2do > 0) {
                break;
            }
        }
        if self.heap.gc.phase == Phase::Pause {
            self.set_pause();
        } else {
            self.set_debt(stepsize);
        }
    }

    fn single_step(&mut self, fast: bool) -> StepResult {
        self.note_running_native();
        match self.heap.gc.phase {
            Phase::Pause => {
                self.restart_collection();
                self.heap.gc.phase = Phase::Propagate;
                StepResult::Work(1)
            }
            Phase::Propagate => {
                if fast || self.heap.gc.gray.is_empty() {
                    self.heap.gc.phase = Phase::EnterAtomic;
                    StepResult::Work(1)
                } else {
                    StepResult::Work(self.heap.propagate_one())
                }
            }
            Phase::EnterAtomic => self.enter_atomic(),
            Phase::SweepAllgc => self.sweep_all(fast),
            Phase::SweepFinobj => self.sweep_finobj(fast),
            Phase::SweepTobefnz => self.sweep_tobefnz(fast),
            Phase::SweepEnd => {
                self.heap.check_string_table_size();
                self.heap.gc.phase = Phase::CallFinalizers;
                StepResult::Work(SWEEP_MAX as i64)
            }
            Phase::CallFinalizers => self.call_one_finalizer(),
            Phase::Atomic => StepResult::Work(0),
        }
    }

    fn sweep_budget(fast: bool) -> usize {
        if fast { usize::MAX } else { SWEEP_MAX }
    }

    fn sweep_all(&mut self, fast: bool) -> StepResult {
        if self.heap.arenas_swept() {
            self.heap.gc.phase = Phase::SweepFinobj;
            self.heap.gc.sweep_index = 0;
        } else {
            self.heap.sweep_arenas(Self::sweep_budget(fast));
        }
        StepResult::Work(SWEEP_MAX as i64)
    }

    fn sweep_finobj(&mut self, fast: bool) -> StepResult {
        if self.heap.list_swept(false) {
            self.heap.gc.phase = Phase::SweepTobefnz;
            self.heap.gc.sweep_index = 0;
        } else {
            self.heap.sweep_list(false, Self::sweep_budget(fast));
        }
        StepResult::Work(SWEEP_MAX as i64)
    }

    fn sweep_tobefnz(&mut self, fast: bool) -> StepResult {
        if self.heap.list_swept(true) {
            self.heap.gc.phase = Phase::SweepEnd;
        } else {
            self.heap.sweep_list(true, Self::sweep_budget(fast));
        }
        StepResult::Work(SWEEP_MAX as i64)
    }

    fn call_one_finalizer(&mut self) -> StepResult {
        if self.heap.gc.tobefnz.is_empty() {
            self.heap.gc.phase = Phase::Pause;
            return StepResult::Step2Pause;
        }
        self.call_next_finalizer();
        StepResult::Work(CWUFIN)
    }

    fn run_to_phase(&mut self, phase: Phase, fast: bool) {
        while self.heap.gc.phase != phase {
            self.single_step(fast);
        }
    }

    fn enter_atomic(&mut self) -> StepResult {
        self.atomic();
        if self.check_major_minor() {
            return StepResult::Step2Minor;
        }
        self.enter_sweep();
        StepResult::AtomicStep
    }

    fn enter_sweep(&mut self) {
        self.heap.gc.phase = Phase::SweepAllgc;
        self.heap.gc.sweep_arena = 0;
        self.heap.gc.sweep_index = 0;
    }

    fn restart_collection(&mut self) {
        self.heap.gc.clear_gray_lists();
        self.heap.gc.marked = 0;
        let roots = self.gc_roots();
        for &root in &roots {
            self.heap.mark_value(root);
        }
        self.heap.gc.roots = roots;
        self.heap.mark_being_finalized();
    }

    fn atomic(&mut self) {
        self.note_running_native();
        let roots = self.gc_roots();
        self.heap.atomic(&roots);
        self.heap.gc.roots = roots;
    }

    fn check_major_minor(&mut self) -> bool {
        if self.heap.gc.kind == Kind::GenMajor {
            let total = self.heap.gc.total;
            let added = total - self.heap.gc.major_minor;
            let limit = if added < 0 {
                0
            } else {
                self.heap.gc.param(Param::MajorMinor, added)
            };
            let tobecollected = total - self.heap.gc.marked;
            if tobecollected > limit {
                self.atomic_to_generational();
                self.set_minor_debt();
                return true;
            }
        }
        self.heap.gc.major_minor = self.heap.gc.marked;
        false
    }

    fn check_minor_major(&self) -> bool {
        let limit = self.heap.gc.param(Param::MinorMajor, self.heap.gc.major_minor);
        if limit == 0 {
            return false;
        }
        self.heap.gc.marked >= limit
    }

    fn minor_to_incremental(&mut self, kind: Kind) {
        self.heap.gc.major_minor = self.heap.gc.marked;
        self.heap.gc.kind = kind;
        self.heap.gc.young.clear();
        self.heap.gc.old1.clear();
        self.enter_sweep();
        let debt = self.heap.gc.param(Param::StepSize, 100);
        self.set_debt(debt);
    }

    fn atomic_to_generational(&mut self) {
        self.heap.gc.clear_gray_lists();
        self.heap.gc.phase = Phase::SweepAllgc;
        self.heap.sweep_all_to_old();
        self.heap.gc.young.clear();
        self.heap.gc.old1.clear();
        self.heap.gc.kind = Kind::GenMinor;
        self.heap.gc.major_minor = self.heap.gc.marked;
        self.heap.gc.marked = 0;
        self.finish_generational_cycle();
    }

    fn enter_generational(&mut self) {
        self.run_to_phase(Phase::Pause, true);
        self.run_to_phase(Phase::Propagate, true);
        self.atomic();
        self.atomic_to_generational();
        self.set_minor_debt();
    }

    fn full_generational(&mut self) {
        self.minor_to_incremental(Kind::Inc);
        self.enter_generational();
    }

    fn full_incremental(&mut self) {
        if self.heap.gc.keeps_invariant() {
            self.enter_sweep();
        }
        self.run_to_phase(Phase::Pause, true);
        self.run_to_phase(Phase::CallFinalizers, true);
        self.run_to_phase(Phase::Pause, true);
        self.set_pause();
    }

    fn young_collection(&mut self) {
        let marked = self.heap.gc.marked;
        let mut old1 = std::mem::take(&mut self.heap.gc.old1);
        self.heap.mark_old_list(&old1);
        old1.clear();
        self.heap.gc.old1 = old1;
        let finobj = std::mem::take(&mut self.heap.gc.finobj);
        self.heap.mark_old_list(&finobj);
        self.heap.gc.finobj = finobj;
        let pending = std::mem::take(&mut self.heap.gc.tobefnz);
        let (front, back) = pending.as_slices();
        self.heap.mark_old_list(front);
        self.heap.mark_old_list(back);
        self.heap.gc.tobefnz = pending;
        self.atomic();
        self.heap.gc.phase = Phase::SweepAllgc;
        let spare = std::mem::take(&mut self.heap.gc.spare_young);
        let young = std::mem::replace(&mut self.heap.gc.young, spare);
        let mut added = self.heap.sweep_young(young);
        let finobj = std::mem::take(&mut self.heap.gc.finobj);
        let (finobj, finobj_added) = self.heap.age_young_list(finobj);
        self.heap.gc.finobj = finobj;
        added += finobj_added;
        let pending: Vec<Gc> = self.heap.gc.tobefnz.drain(..).collect();
        let (pending, pending_added) = self.heap.age_young_list(pending);
        self.heap.gc.tobefnz = pending.into();
        added += pending_added;
        self.heap.gc.marked = marked + added;
        if self.check_minor_major() {
            self.minor_to_incremental(Kind::GenMajor);
            self.heap.gc.marked = 0;
        } else {
            self.finish_generational_cycle();
        }
    }

    fn finish_generational_cycle(&mut self) {
        self.heap.correct_gray_lists();
        self.heap.check_string_table_size();
        self.heap.gc.phase = Phase::Propagate;
        if !self.heap.gc.tobefnz.is_empty() {
            self.call_all_pending_finalizers();
        }
    }

    pub(crate) fn check_finalizer(&mut self, object: Gc, metatable: Option<Ref<Table>>) {
        let Some(marks) = self.heap.marks_of(object) else {
            return;
        };
        if marks & FINALIZED != 0 || self.heap.gc.stop & STOP_CLOSING != 0 {
            return;
        }
        let Some(metatable) = metatable else {
            return;
        };
        if self.table_event(metatable, Event::Gc).is_nil() {
            return;
        }
        let marks = if self.heap.gc.in_sweep() {
            make_white(marks, self.heap.gc.current_white)
        } else {
            marks
        };
        self.heap.set_marks_of(object, marks | FINALIZED);
        self.heap.gc.finobj.push(object);
    }

    pub fn close(&mut self) {
        let switch = self.switch_into(self.main_thread);
        self.close_pending_at_exit();
        self.heap.gc.stop = STOP_CLOSING;
        self.change_mode(Kind::Inc);
        self.heap.separate_to_be_finalized(true);
        self.call_all_pending_finalizers();
        self.switch_back(switch);
    }

    fn call_all_pending_finalizers(&mut self) {
        while !self.heap.gc.tobefnz.is_empty() {
            self.call_next_finalizer();
        }
    }

    fn call_next_finalizer(&mut self) {
        let Some(object) = self.heap.gc.tobefnz.pop_front() else {
            return;
        };
        self.resurrect(object);
        let Some(value) = gc_value(object) else {
            return;
        };
        let handler = self.metamethod(value, Event::Gc);
        if handler.is_nil() {
            return;
        }
        let saved_stop = self.heap.gc.stop;
        self.heap.gc.stop |= STOP_COLLECTOR;
        self.heap.gc.finalizers_called = self.heap.gc.finalizers_called.wrapping_add(1);
        let outcome = self.call_protected_as(handler, &[value], None, true);
        self.heap.gc.stop = saved_stop;
        if let Err(error) = outcome {
            self.warn_finalizer_error(error);
        }
    }

    fn warn_finalizer_error(&mut self, error: LuaError) {
        let message = match error.value() {
            Value::String(text) => self.heap.bytes(text).to_vec(),
            _ => b"error object is not a string".to_vec(),
        };
        self.warning(b"error in __gc (", true);
        self.warning(&message, true);
        self.warning(b")", false);
    }

    fn resurrect(&mut self, object: Gc) {
        let Some(marks) = self.heap.marks_of(object) else {
            return;
        };
        let marks = marks & !FINALIZED;
        let in_sweep = self.heap.gc.in_sweep();
        let marks = if in_sweep {
            make_white(marks, self.heap.gc.current_white)
        } else {
            marks
        };
        self.heap.set_marks_of(object, marks);
        if !in_sweep && age(marks) == OLD1 {
            self.heap.gc.old1.push(object);
        }
    }

    /// The caller returns the buffer to `GcState::roots`.
    fn gc_roots(&mut self) -> Vec<Value> {
        let mut roots = std::mem::take(&mut self.heap.gc.roots);
        roots.clear();
        roots.extend([
            Value::Table(self.globals),
            Value::Table(self.registry),
            Value::Thread(self.main_thread),
            Value::Thread(self.current_thread),
        ]);
        roots.extend(self.type_metatables.iter().flatten().map(|table| Value::Table(*table)));
        roots.extend(self.string_metatable.map(Value::Table));
        roots.extend(self.current_native.map(|call| Value::Native(call.native)));
        roots.extend(self.hidden.iter().map(|(_, value)| *value));
        roots.extend(self.yielded.iter().copied());
        roots.extend(self.pins.iter().copied());
        roots.extend(self.heap.events.iter().map(|name| Value::String(*name)));
        roots
    }

    pub(crate) fn pin(&mut self, values: impl IntoIterator<Item = Value>) -> usize {
        let mark = self.pins.len();
        self.pins.extend(values);
        mark
    }

    pub(crate) fn unpin(&mut self, mark: usize) {
        self.pins.truncate(mark);
    }

    pub(crate) fn gc_collect(&mut self) {
        match self.heap.gc.kind {
            Kind::GenMinor => self.full_generational(),
            Kind::Inc => self.full_incremental(),
            Kind::GenMajor => {
                self.heap.gc.kind = Kind::Inc;
                self.full_incremental();
                self.heap.gc.kind = Kind::GenMajor;
            }
        }
    }

    pub(crate) fn gc_total_bytes(&self) -> i64 {
        self.heap.gc.total
    }

    pub(crate) fn gc_step_by(&mut self, bytes: i64) -> bool {
        let saved_stop = self.heap.gc.stop;
        self.heap.gc.stop = 0;
        let debt = self.heap.gc.debt;
        let new_debt = if bytes <= 0 {
            0
        } else if debt >= bytes - MAX_LMEM {
            debt - bytes
        } else {
            -MAX_LMEM
        };
        self.set_debt(new_debt);
        let mut worked = false;
        if self.heap.gc.debt <= 0 {
            self.gc_step();
            worked = true;
        }
        self.heap.gc.stop = saved_stop;
        worked && self.heap.gc.phase == Phase::Pause
    }

    pub(crate) fn gc_param(&mut self, param: Param, value: i64) -> i64 {
        let slot = param.index();
        let previous = apply_param(self.heap.gc.params[slot], 100) as i32 as i64;
        let value = value as i32;
        if value >= 0 {
            self.heap.gc.params[slot] = code_param(value as u32);
        }
        previous
    }

    pub(crate) fn gc_change_mode(&mut self, kind: Kind) -> bool {
        let previous = self.heap.gc.is_generational_mode();
        self.change_mode(kind);
        previous
    }

    fn change_mode(&mut self, kind: Kind) {
        if self.heap.gc.kind == Kind::GenMajor {
            self.heap.gc.kind = Kind::Inc;
        }
        if kind == self.heap.gc.kind {
            return;
        }
        if kind == Kind::Inc {
            self.minor_to_incremental(Kind::Inc);
        } else {
            self.enter_generational();
        }
    }

    pub fn use_generational_collector(&mut self) {
        self.change_mode(Kind::GenMinor);
    }

    pub(crate) fn gc_stop(&mut self) {
        self.heap.gc.stop = STOP_USER;
    }

    pub(crate) fn gc_restart(&mut self) {
        self.set_debt(0);
        self.heap.gc.stop = 0;
    }

    pub(crate) fn gc_is_running(&self) -> bool {
        self.heap.gc.running()
    }

    pub(crate) fn pin_resume(&mut self, resume: &crate::object::Resume, owner: Value) -> usize {
        let mut values = values![owner];
        resume.values(&mut values);
        self.pin(values)
    }

    pub(crate) fn set_table_metatable_checked(&mut self, table: Ref<Table>, metatable: Option<Ref<Table>>) {
        self.heap.set_table_metatable(table, metatable);
        self.check_finalizer(Gc::Table(table), metatable);
    }

    pub(crate) fn set_userdata_metatable_checked(
        &mut self,
        userdata: Ref<crate::object::Userdata>,
        metatable: Option<Ref<Table>>,
    ) {
        self.heap.set_userdata_metatable(userdata, metatable);
        self.check_finalizer(Gc::Userdata(userdata), metatable);
    }

    pub(crate) fn gc_stopped_by_collector(&self) -> bool {
        self.heap.gc.stop & STOP_COLLECTOR != 0
    }
}
