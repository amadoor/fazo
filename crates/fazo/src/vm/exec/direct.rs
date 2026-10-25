use super::{Access, Activation, Flow, MAX_ARRAY_SIZE, SETLIST_BLOCK, Store};
use crate::{
    LuaError, Vm,
    arena::Ref,
    instruction::Instruction,
    object::Closure,
    proto::Proto,
    value::Value,
    vm::{fast, naming::Operand},
};

pub(crate) enum Direct {
    Next(usize),
    Reload,
    Unsupported,
}

impl Vm {
    /// Each arm runs the interpreter's arm with its pc after the fetch, so saved pcs and metamethod continuations match.
    pub(crate) fn run_direct(&mut self, closure: Ref<Closure>, base: usize, proto: Ref<Proto>, pc: usize) -> Result<Direct, LuaError> {
        let activation = Activation { closure, base, proto };
        let instruction = self.heap.protos.live(proto).code[pc];
        let mut next = pc + 1;

        let flow = match instruction {
            Instruction::NewTable { a, b, c, k } => {
                let extra = if k { self.extra_argument_at(proto, next)? } else { 0 };
                let array = u64::from(c) + u64::from(extra) * u64::from(SETLIST_BLOCK);
                next += 1;
                self.direct_allocation_point(activation, next, instruction)?;
                let table = self.new_table_sized(array, b, array > 0 && self.heap.protos.live(proto).exec.words.opens_list(pc))?;
                self.set_reg(activation, a, Value::Table(table));
                Flow::Next
            }
            Instruction::Closure { a, bx } => {
                self.direct_allocation_point(activation, next, instruction)?;
                let closure = self.new_closure_in(activation, bx as usize);
                self.set_reg(activation, a, Value::Closure(closure));
                Flow::Next
            }
            Instruction::EqK { a, b, k } => {
                let left = self.reg(activation, a);
                let right = self.heap.protos.live(proto).constants[usize::from(b)];
                let condition = match (left, right) {
                    (Value::Integer(left), Value::Integer(right)) => left == right,
                    _ => self.heap.raw_equal(left, right),
                };
                self.conditional_jump(activation, &mut next, condition, k);
                Flow::Next
            }
            Instruction::Eq { a, b, k } => {
                let (left, right) = (self.reg(activation, a), self.reg(activation, b));
                match fast::equal(left, right) {
                    Some(outcome) => {
                        self.conditional_jump(activation, &mut next, outcome, k);
                        Flow::Next
                    }
                    None => self.equal_slow(activation, &mut next, left, right, k)?,
                }
            }
            Instruction::Lt { a, b, k } => {
                let (left, right) = (self.reg(activation, a), self.reg(activation, b));
                match fast::less_than(left, right) {
                    Some(outcome) => {
                        self.conditional_jump(activation, &mut next, outcome, k);
                        Flow::Next
                    }
                    None => self.less_than_slow(activation, &mut next, left, right, k)?,
                }
            }
            Instruction::Le { a, b, k } => {
                let (left, right) = (self.reg(activation, a), self.reg(activation, b));
                match fast::less_equal(left, right) {
                    Some(outcome) => {
                        self.conditional_jump(activation, &mut next, outcome, k);
                        Flow::Next
                    }
                    None => self.less_equal_slow(activation, &mut next, left, right, k)?,
                }
            }
            Instruction::Concat { a, b } => {
                self.direct_allocation_point(activation, next, instruction)?;
                self.concat_slow(activation, next, a, b)?
            }
            Instruction::Close { a } => self.close_slow(activation, &mut next, a)?,
            Instruction::SetList { a, b, c, k } => {
                let mut stored = c;
                if k {
                    let extra = self.extra_argument_at(proto, next)?;
                    let last = u64::from(stored) + u64::from(extra) * u64::from(SETLIST_BLOCK);
                    if last > MAX_ARRAY_SIZE {
                        return Err(self.memory_error());
                    }
                    stored = last as u32;
                    next += 1;
                }
                self.set_list(activation, a, b, stored)?;
                Flow::Next
            }
            Instruction::Vararg { a, b, c, k } => {
                self.copy_varargs(activation, a, c, k.then_some(b))?;
                Flow::Next
            }
            Instruction::TailCall { a, b, k, .. } => self.tail_call_slow(activation, next, a, b, k)?,
            Instruction::Call { a, b, c } => self.call_slow(activation, next, a, b, c)?,
            Instruction::TForPrep { a, bx } => self.tfor_prep_slow(activation, &mut next, a, bx)?,
            Instruction::TForCall { a, c } => self.generic_call(activation, next, a, c)?,
            Instruction::VarargPrep { .. } => {
                self.adjust_varargs(activation)?;
                if self.heap.protos.live(proto).vararg_table {
                    Flow::Next
                } else {
                    self.save_pc(next);
                    Flow::Reload
                }
            }
            Instruction::Return { a, b, k, .. } => self.return_slow(activation, &mut next, a, b, k)?,
            Instruction::SetTabUp { a, b, c, k } => {
                let object = self.upvalue_get(self.upvalue_ref(activation, a));
                let constants = &self.heap.protos.live(proto).constants;
                let key = constants[usize::from(b)];
                let value = if k { constants[usize::from(c)] } else { self.reg(activation, c) };
                if self.store_field(proto, pc, object, key, value) {
                    Flow::Next
                } else {
                    let access = Access { object, key, operand: Operand::Upvalue(a) };
                    self.set_slow(next, Store { access, value })?
                }
            }
            Instruction::GetVarg { a, c, .. } => {
                self.get_vararg(activation, a, c);
                Flow::Next
            }
            Instruction::Tbc { a } => self.tbc_slow(next, a)?,
            Instruction::ErrNil { a, bx } => self.err_nil(activation, next, a, bx)?,
            _ => return Ok(Direct::Unsupported),
        };

        Ok(match flow {
            Flow::Next | Flow::Done | Flow::Untraced => Direct::Next(next),
            Flow::Reload => Direct::Reload,
        })
    }

    /// The caller has checked that an `EXTRAARG` follows `NEWTABLE`.
    pub(crate) fn direct_new_table(&mut self, closure: Ref<Closure>, base: usize, proto: Ref<Proto>, pc: usize, sizes: (u8, u64, u8)) -> Result<usize, LuaError> {
        let activation = Activation { closure, base, proto };
        let instruction = self.heap.protos.live(proto).code[pc];
        let (a, array, hash_log) = sizes;
        let next = pc + 2;

        self.direct_allocation_point(activation, next, instruction)?;
        let table = self.new_table_sized(array, hash_log, array > 0 && self.heap.protos.live(proto).exec.words.opens_list(pc))?;
        self.set_reg(activation, a, Value::Table(table));
        Ok(next)
    }

    pub(crate) fn direct_closure(&mut self, closure: Ref<Closure>, base: usize, proto: Ref<Proto>, pc: usize, (a, index): (u8, usize)) -> Result<usize, LuaError> {
        let activation = Activation { closure, base, proto };
        let instruction = self.heap.protos.live(proto).code[pc];
        let next = pc + 1;

        self.direct_allocation_point(activation, next, instruction)?;
        let created = self.new_closure_in(activation, index);
        self.set_reg(activation, a, Value::Closure(created));
        Ok(next)
    }

    /// The interpreter's `allocation_point`, without refreshing a register window.
    #[inline(always)]
    fn direct_allocation_point(&mut self, activation: Activation, pc: usize, instruction: Instruction) -> Result<(), LuaError> {
        if self.heap.gc.debt > 0 {
            return Ok(());
        }
        self.direct_collect(activation, pc, instruction)
    }

    #[cold]
    #[inline(never)]
    fn direct_collect(&mut self, activation: Activation, pc: usize, instruction: Instruction) -> Result<(), LuaError> {
        self.save_pc(pc);
        self.collect_before(activation, instruction);
        self.check_memory()
    }

    fn extra_argument_at(&mut self, proto: Ref<Proto>, pc: usize) -> Result<u32, LuaError> {
        match self.heap.protos.live(proto).code.get(pc) {
            Some(&Instruction::ExtraArg { ax }) => Ok(ax),
            _ => Err(self.runtime_error(b"invalid instruction")),
        }
    }
}
