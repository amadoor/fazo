//! Follows V8 Maglev's `MaglevGraphBuilder` (src/maglev/maglev-graph-builder.cc).
//! Registers live in SSA values until code that may observe the frame.

use super::{
    flow::{self, Flow, RegSet},
    ir::{BinOp, BlockId, Cc, Closure, Const, Counted, Edge, FrameState, Func, Helper, HelperKind, InlineFrame, Loop, NumOp, Op, PureFn, Rep, StateId, Term, Type, UnOp, Val},
    passes::inline::{self, Candidate},
};
use crate::{
    arena::Ref,
    instruction::Instruction,
    jit::sites,
    proto::Proto,
    value::Value as LuaValue,
    vm::{
        Event,
        ic::{ABSENT, Entry, InlineCache, MethodEntry},
    },
};

pub(crate) struct Facts<'a> {
    pub(crate) inline_caches: bool,
    pub(crate) avoid: &'a [u32],
    pub(crate) interned: &'a dyn Fn(LuaValue) -> bool,
    pub(crate) proto_of: &'a dyn Fn(Ref<Proto>) -> Option<&'a Proto>,
    pub(crate) policy: inline::Policy,
    pub(crate) osr_at: Option<u32>,
    /// Per pc of the compiled prototype: where the running closure's environment table held the global's key.
    pub(crate) nodes: &'a [Option<NodeHint>],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct NodeHint {
    pub(crate) node: u32,
    pub(crate) nodes: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reg {
    Mem,
    Val(Val, bool),
}

struct Incoming {
    from: BlockId,
    edge: usize,
    regs: Vec<Reg>,
}

struct Pending {
    target: usize,
    incoming: Incoming,
}

struct LoopBuild {
    header: BlockId,
    params: Vec<(u32, Val, bool)>,
    index: usize,
}


struct Frame<'a> {
    proto: &'a Proto,
    base: u32,
    chain: Vec<InlineFrame>,
    site: Option<u32>,
    flow: Flow,
    captured: RegSet,
    pending: Vec<Pending>,
    loops: Vec<Option<LoopBuild>>,
    counted: Vec<Option<i64>>,
    inline_prep: Vec<bool>,
    returns: Vec<Incoming>,
    caller: Option<(u32, i32)>,
}

pub(crate) fn build<'a>(proto: &'a Proto, facts: &Facts<'a>) -> Func {
    let mut builder = Builder {
        func: Func::default(),
        facts,
        regs: Vec::new(),
        current: None,
    };
    let entry = builder.func.new_block();
    builder.func.entry = Some(entry);
    builder.current = Some(entry);
    let mut frame = builder.frame(proto, 0, Vec::new(), None, None);
    builder.regs = vec![Reg::Mem; usize::from(proto.max_stack).max(1)];
    builder.jump_to(&mut frame, 0);
    builder.run(&mut frame);
    builder.func
}

struct Builder<'a, 'f> {
    func: Func,
    facts: &'f Facts<'a>,
    regs: Vec<Reg>,
    current: Option<BlockId>,
}

impl<'a> Builder<'a, '_> {
    fn frame(&self, proto: &'a Proto, base: u32, chain: Vec<InlineFrame>, site: Option<u32>, caller: Option<(u32, i32)>) -> Frame<'a> {
        let captured = if chain.is_empty() {
            flow::captured_registers(proto, &|index| {
                let child = (self.facts.proto_of)(*proto.protos.get(index)?)?;
                Some(child.upvalues.iter().map(|upvalue| (upvalue.in_stack, upvalue.index)).collect())
            })
        } else {
            RegSet::default()
        };
        let flow = flow::analyze(proto, &captured);
        let length = proto.code.len();
        Frame {
            proto,
            base,
            chain,
            site,
            flow,
            captured,
            pending: Vec::new(),
            loops: (0..length).map(|_| None).collect(),
            counted: vec![None; length],
            inline_prep: vec![false; length],
            returns: Vec::new(),
            caller,
        }
    }

    fn run(&mut self, frame: &mut Frame<'a>) {
        let mut pc = 0;
        while pc < frame.proto.code.len() {
            if frame.flow.leaders[pc] || frame.pending.iter().any(|pending| pending.target == pc) {
                self.start_block(frame, pc);
            }
            if self.current.is_none() {
                pc += 1;
                continue;
            }
            pc = self.translate(frame, pc);
        }
    }

    fn block(&self) -> BlockId {
        self.current.expect("translation runs inside a block")
    }

    fn emit(&mut self, op: Op, ty: Type, rep: Rep) -> Val {
        let block = self.block();
        self.func.append(block, op, ty, rep)
    }

    fn effect(&mut self, op: Op) {
        self.emit(op, Type::NONE, Rep::None);
    }

    fn constant(&mut self, constant: Const) -> Val {
        let rep = match constant {
            Const::Nil | Const::Gc { tag: 10, .. } => Rep::Boxed,
            Const::Bool(_) => Rep::Bool,
            Const::Int(_) => Rep::Int,
            Const::Float(_) => Rep::Float,
            Const::Gc { .. } => Rep::Payload,
        };
        self.emit(Op::Const(constant), constant.ty(), rep)
    }

    fn slot(frame: &Frame<'_>, register: u8) -> u32 {
        frame.base + u32::from(register)
    }

    fn ensure_slot(&mut self, slot: u32) {
        if self.regs.len() <= slot as usize {
            self.regs.resize(slot as usize + 1, Reg::Mem);
        }
    }

    fn read(&mut self, frame: &Frame<'_>, register: u8) -> Val {
        let slot = Self::slot(frame, register);
        self.ensure_slot(slot);
        if let Reg::Val(value, _) = self.regs[slot as usize] {
            return value;
        }
        let value = self.emit(Op::LoadReg { slot }, Type::ANY, Rep::Boxed);
        if !frame.captured.contains(usize::from(register)) {
            self.regs[slot as usize] = Reg::Val(value, false);
        }
        value
    }

    fn write(&mut self, frame: &Frame<'_>, register: u8, value: Val) {
        let slot = Self::slot(frame, register);
        self.ensure_slot(slot);
        if frame.captured.contains(usize::from(register)) {
            self.effect(Op::StoreReg { slot, value });
            self.regs[slot as usize] = Reg::Mem;
            return;
        }
        self.regs[slot as usize] = Reg::Val(value, true);
    }

    fn state_at(&mut self, frame: &Frame<'_>, pc: usize) -> StateId {
        let slots = self
            .regs
            .iter()
            .enumerate()
            .filter_map(|(slot, reg)| match *reg {
                Reg::Val(value, true) => Some((slot as u32, value)),
                _ => None,
            })
            .collect();
        let state = FrameState {
            pc: pc as u32,
            frames: frame.chain.clone(),
            slots,
            site: frame.site.unwrap_or(pc as u32),
        };
        self.func.add_state(state)
    }

    fn flush(&mut self) {
        for slot in 0..self.regs.len() {
            if let Reg::Val(value, true) = self.regs[slot] {
                self.effect(Op::StoreReg { slot: slot as u32, value });
                self.regs[slot] = Reg::Val(value, false);
            }
        }
    }

    fn forget(&mut self) {
        self.regs.iter_mut().for_each(|reg| *reg = Reg::Mem);
    }

    fn speculates(&self, frame: &Frame<'_>, pc: usize) -> bool {
        frame.site.is_some() || !self.facts.avoid.contains(&(pc as u32))
    }

    fn end(&mut self, term: Term) {
        let block = self.block();
        self.func.block_mut(block).term = term;
        self.current = None;
    }

    fn jump_to(&mut self, frame: &mut Frame<'a>, target: usize) {
        let from = self.block();
        self.end(Term::Jump(Edge::to(BlockId(u32::MAX))));
        self.edge_to(frame, from, 0, target);
    }

    fn edge_to(&mut self, frame: &mut Frame<'a>, from: BlockId, edge: usize, target: usize) {
        if target <= frame.flow.successors.len() && frame.loops.get(target).is_some_and(Option::is_some) {
            self.back_edge(frame, from, edge, target);
            return;
        }
        frame.pending.push(Pending {
            target,
            incoming: Incoming { from, edge, regs: self.regs.clone() },
        });
    }

    fn branch_to(&mut self, frame: &mut Frame<'a>, cond: Val, then: usize, other: usize) {
        let from = self.block();
        let regs = self.regs.clone();
        self.end(Term::Branch {
            cond,
            then: Edge::to(BlockId(u32::MAX)),
            other: Edge::to(BlockId(u32::MAX)),
        });
        self.edge_to(frame, from, 0, then);
        self.regs = regs;
        self.edge_to(frame, from, 1, other);
    }

    fn set_edge(&mut self, from: BlockId, edge: usize, block: BlockId, args: Vec<Val>) {
        let mut edges = self.func.block_mut(from).term.edges_mut();
        let target = &mut edges[edge];
        target.block = block;
        target.args = args;
    }

    fn live(&self, frame: &Frame<'_>, pc: usize, slot: u32) -> bool {
        if slot < frame.base {
            return true;
        }
        let register = (slot - frame.base) as usize;
        register < usize::from(frame.proto.max_stack) && frame.flow.live[pc].contains(register)
    }

    fn start_block(&mut self, frame: &mut Frame<'a>, pc: usize) {
        if let Some(current) = self.current {
            let regs = self.regs.clone();
            self.func.block_mut(current).term = Term::Jump(Edge::to(BlockId(u32::MAX)));
            frame.pending.push(Pending {
                target: pc,
                incoming: Incoming { from: current, edge: 0, regs },
            });
            self.current = None;
        }
        let mut incoming = Vec::new();
        let mut index = 0;
        while index < frame.pending.len() {
            if frame.pending[index].target == pc {
                incoming.push(frame.pending.swap_remove(index).incoming);
            } else {
                index += 1;
            }
        }
        let osr = frame.flow.headers[pc] && frame.chain.is_empty() && (incoming.is_empty() || self.enters_loop(frame, pc));
        if incoming.is_empty() && !osr {
            return;
        }
        if frame.flow.headers[pc] {
            self.start_loop(frame, pc, incoming);
            return;
        }
        let block = self.merge(frame, pc, incoming);
        self.current = Some(block);
    }

    /// Follows V8 Maglev's MergeIntoFrameState (src/maglev/maglev-interpreter-frame-state.cc).
    fn merge(&mut self, frame: &Frame<'a>, pc: usize, incoming: Vec<Incoming>) -> BlockId {
        let block = self.func.new_block();
        let width = incoming.iter().map(|edge| edge.regs.len()).max().unwrap_or(0).max(self.regs.len());
        let mut merged = vec![Reg::Mem; width];
        let mut args: Vec<Vec<Val>> = vec![Vec::new(); incoming.len()];
        for (slot, merged_reg) in merged.iter_mut().enumerate() {
            let states: Vec<Reg> = incoming.iter().map(|edge| edge.regs.get(slot).copied().unwrap_or(Reg::Mem)).collect();
            if states.iter().all(|state| *state == states[0]) {
                *merged_reg = states[0];
                continue;
            }
            if !self.live(frame, pc, slot as u32) || states.iter().all(|state| *state == Reg::Mem) {
                self.store_dirty(&incoming, &states, slot as u32, frame, pc);
                continue;
            }
            let ty = self.union_type(&states);
            let dirty = states.iter().any(|state| matches!(state, Reg::Val(_, true)));
            let param = self.func.add_param(block, ty, Rep::Boxed);
            for (index, state) in states.iter().enumerate() {
                let value = self.value_in(incoming[index].from, *state, slot as u32);
                args[index].push(value);
            }
            *merged_reg = Reg::Val(param, dirty);
        }
        for (edge, values) in incoming.iter().zip(args) {
            self.set_edge(edge.from, edge.edge, block, values);
        }
        self.regs = merged;
        block
    }

    /// A dead register is dropped at a merge, except a named local in scope, which `debug.getlocal` reads.
    fn store_dirty(&mut self, incoming: &[Incoming], states: &[Reg], slot: u32, frame: &Frame<'_>, pc: usize) {
        let named = slot >= frame.base && ((slot - frame.base) as usize) < flow::active_locals(frame.proto, pc);
        if !named {
            return;
        }
        for (edge, state) in incoming.iter().zip(states) {
            if let Reg::Val(value, true) = *state {
                self.func.append(edge.from, Op::StoreReg { slot, value }, Type::NONE, Rep::None);
            }
        }
    }

    fn union_type(&self, states: &[Reg]) -> Type {
        states.iter().fold(Type::NONE, |ty, state| match state {
            Reg::Val(value, _) => ty.union(self.func.ty(*value)),
            Reg::Mem => Type::ANY,
        })
    }

    fn value_in(&mut self, block: BlockId, state: Reg, slot: u32) -> Val {
        match state {
            Reg::Val(value, _) => value,
            Reg::Mem => self.func.append(block, Op::LoadReg { slot }, Type::ANY, Rep::Boxed),
        }
    }

    /// Only root-frame headers get an on-stack-replacement entry, since the interpreter enters only root frames.
    /// A header no forward edge reaches, such as a generic `for` loop's body, which only its back edge enters, keeps its entry.
    fn start_loop(&mut self, frame: &mut Frame<'a>, pc: usize, mut incoming: Vec<Incoming>) {
        if frame.chain.is_empty() && (incoming.is_empty() || self.enters_loop(frame, pc)) {
            let osr = self.osr_entry(frame, pc, &incoming);
            incoming.push(osr);
        }
        let preheader = self.merge(frame, pc, incoming);
        self.current = Some(preheader);
        let entry_state = self.state_at(frame, pc);
        let header = self.func.new_block();
        let mut params = Vec::new();
        let mut args = Vec::new();
        for slot in 0..self.regs.len() as u32 {
            if !self.live(frame, pc, slot) {
                continue;
            }
            let (value, dirty) = match self.regs[slot as usize] {
                Reg::Val(value, dirty) => (value, dirty),
                Reg::Mem if slot >= frame.base && frame.captured.contains((slot - frame.base) as usize) => continue,
                Reg::Mem => (self.emit(Op::LoadReg { slot }, Type::ANY, Rep::Boxed), false),
            };
            let param = self.func.add_param(header, Type::ANY, Rep::Boxed);
            params.push((slot, param, dirty));
            args.push(value);
            self.regs[slot as usize] = Reg::Val(param, dirty);
        }
        self.func.block_mut(preheader).term = Term::Jump(Edge { block: header, args });
        let counted = frame.counted[pc].and_then(|step| {
            let a = self.for_base_slot(frame, pc)?;
            let param = |wanted: u32| params.iter().find(|&&(slot, _, _)| slot == wanted).map(|&(_, param, _)| param);
            Some(Counted { index: param(a + 2)?, count: param(a)?, step })
        });
        self.func.loops.push(Loop {
            header,
            preheader,
            blocks: Vec::new(),
            entry_state,
            counted,
        });
        frame.loops[pc] = Some(LoopBuild { header, params, index: self.func.loops.len() - 1 });
        self.current = Some(header);
    }

    /// Maglev builds one on-stack-replacement entry, at the loop being run (`entrypoint_` from `osr_offset`, src/maglev/maglev-graph-builder.cc); this keeps entries for the loops around the tier-up point, so an exit inside any of them can come back, or for every loop when the point is unknown.
    fn enters_loop(&self, frame: &Frame<'_>, header: usize) -> bool {
        let Some(at) = self.facts.osr_at else {
            return true;
        };
        let at = at as usize;
        header <= at && frame.flow.successors.iter().skip(at).any(|targets| targets.contains(&header))
    }

    fn for_base_slot(&self, frame: &Frame<'_>, header: usize) -> Option<u32> {
        let prep = header.checked_sub(1)?;
        match frame.proto.code.get(prep)? {
            Instruction::ForPrep { a, .. } => Some(Self::slot(frame, *a)),
            _ => None,
        }
    }

    /// Follows Maglev's speculation on OSR values (`kSpeculativeOSRValue`, src/maglev/maglev-phi-representation-selector.h).
    fn osr_entry(&mut self, frame: &Frame<'a>, pc: usize, incoming: &[Incoming]) -> Incoming {
        let block = self.func.new_block();
        self.func.osr.push((pc as u32, block));
        let width = self.regs.len();
        let saved = std::mem::replace(&mut self.regs, vec![Reg::Mem; width]);
        self.current = Some(block);
        let state = self.state_at(frame, pc);
        let speculate = self.speculates(frame, pc);
        for slot in 0..self.regs.len() as u32 {
            if !self.live(frame, pc, slot) || frame.captured.contains(slot as usize) {
                continue;
            }
            let kinds: Vec<Type> = incoming.iter().map(|edge| match edge.regs.get(slot as usize) {
                Some(Reg::Val(value, _)) => self.entry_type(frame, *value),
                _ => Type::ANY,
            }).collect();
            let ty = kinds.iter().fold(Type::NONE, |ty, kind| ty.union(*kind));
            let loaded = self.emit(Op::LoadReg { slot }, Type::ANY, Rep::Boxed);
            let value = if speculate && !kinds.is_empty() && (ty == Type::INT || ty == Type::FLOAT) {
                self.emit(Op::GuardType { value: loaded, ty, state }, ty, Rep::Boxed)
            } else {
                loaded
            };
            self.regs[slot as usize] = Reg::Val(value, false);
        }
        self.end(Term::Jump(Edge::to(BlockId(u32::MAX))));
        let regs = std::mem::replace(&mut self.regs, saved);
        Incoming { from: block, edge: 0, regs }
    }

    /// Header parameters are untyped until the type pass, so speculation reads the type the loop is entered with.
    fn entry_type(&self, frame: &Frame<'_>, value: Val) -> Type {
        let mut value = value;
        while let Some(arg) = self.preheader_arg(frame, value) {
            value = arg;
        }
        self.func.ty(value)
    }

    fn preheader_arg(&self, frame: &Frame<'_>, value: Val) -> Option<Val> {
        frame.loops.iter().flatten().find_map(|build| {
            let index = self.func.block(build.header).params.iter().position(|&param| param == value)?;
            let preheader = self.func.loops[build.index].preheader;
            match &self.func.block(preheader).term {
                Term::Jump(edge) => edge.args.get(index).copied(),
                _ => None,
            }
        })
    }

    fn back_edge(&mut self, frame: &mut Frame<'a>, from: BlockId, edge: usize, target: usize) {
        let Some(build) = frame.loops[target].as_ref() else {
            return;
        };
        let header = build.header;
        let params = build.params.clone();
        let index = build.index;
        let mut args = Vec::with_capacity(params.len());
        for (slot, _, dirty) in params {
            let state = self.regs.get(slot as usize).copied().unwrap_or(Reg::Mem);
            let value = self.value_in(from, state, slot);
            if !dirty && matches!(state, Reg::Val(_, true)) {
                self.func.append(from, Op::StoreReg { slot, value }, Type::NONE, Rep::None);
            }
            args.push(value);
        }
        self.store_unparameterized(frame, from, target);
        self.set_edge(from, edge, header, args);
        let blocks: Vec<BlockId> = (header.0..self.func.blocks.len() as u32).map(BlockId).collect();
        let entry = &mut self.func.loops[index];
        for block in blocks {
            if !entry.blocks.contains(&block) {
                entry.blocks.push(block);
            }
        }
    }

    /// A register the header holds in memory must be in memory again when the loop comes back to the header.
    fn store_unparameterized(&mut self, frame: &Frame<'_>, from: BlockId, target: usize) {
        let Some(build) = frame.loops[target].as_ref() else {
            return;
        };
        for slot in 0..self.regs.len() as u32 {
            if build.params.iter().any(|&(param_slot, _, _)| param_slot == slot) || !self.live(frame, target, slot) {
                continue;
            }
            if let Reg::Val(value, true) = self.regs[slot as usize] {
                self.func.append(from, Op::StoreReg { slot, value }, Type::NONE, Rep::None);
            }
        }
    }

    fn translate(&mut self, frame: &mut Frame<'a>, pc: usize) -> usize {
        let instruction = frame.proto.code[pc];
        let next = pc + flow::width(frame.proto, pc);
        match instruction {
            Instruction::Move { a, b } => {
                let value = self.read(frame, b);
                self.write(frame, a, value);
            }
            Instruction::LoadI { a, sbx } => {
                let value = self.constant(Const::Int(i64::from(sbx)));
                self.write(frame, a, value);
            }
            Instruction::LoadF { a, sbx } => {
                let value = self.constant(Const::Float(f64::from(sbx)));
                self.write(frame, a, value);
            }
            Instruction::LoadK { a, bx } => return self.load_constant(frame, pc, a, bx, next),
            Instruction::LoadKx { a } => match frame.proto.code.get(pc + 1) {
                Some(Instruction::ExtraArg { ax }) => return self.load_constant(frame, pc, a, *ax, next),
                _ => return self.fallback(frame, pc),
            },
            Instruction::LoadFalse { a } | Instruction::LFalseSkip { a } => {
                let value = self.constant(Const::Bool(false));
                self.write(frame, a, value);
                if matches!(instruction, Instruction::LFalseSkip { .. }) {
                    self.jump_to(frame, pc + 2);
                    return pc + 1;
                }
            }
            Instruction::LoadTrue { a } => {
                let value = self.constant(Const::Bool(true));
                self.write(frame, a, value);
            }
            Instruction::LoadNil { a, b } => {
                let value = self.constant(Const::Nil);
                for register in a..=a.saturating_add(b) {
                    self.write(frame, register, value);
                }
            }
            Instruction::GetUpval { a, b } => {
                let closure = self.frame_closure(frame);
                let value = self.emit(Op::LoadUpval { closure, index: b }, Type::ANY, Rep::Boxed);
                self.write(frame, a, value);
            }
            Instruction::SetUpval { a, b } => {
                let closure = self.frame_closure(frame);
                let value = self.read(frame, a);
                self.effect(Op::StoreUpval { closure, index: b, value });
            }
            Instruction::GetTabUp { a, b, c } => {
                let closure = self.frame_closure(frame);
                let object = self.emit(Op::LoadUpval { closure, index: b }, Type::ANY, Rep::Boxed);
                if let Some(node) = self.node_hint(frame, pc, c) {
                    return self.get_node(frame, pc, a, object, node);
                }
                return self.get_field(frame, pc, a, object, c);
            }
            Instruction::GetField { a, b, c } => {
                let object = self.read(frame, b);
                return self.get_field(frame, pc, a, object, c);
            }
            Instruction::GetTable { a, b, c } => {
                let object = self.read(frame, b);
                let key = self.read(frame, c);
                return self.index(frame, pc, a, object, key);
            }
            Instruction::GetI { a, b, c } => {
                let object = self.read(frame, b);
                let key = self.constant(Const::Int(i64::from(c)));
                return self.index(frame, pc, a, object, key);
            }
            Instruction::SelfOp { a, b, c } => return self.self_op(frame, pc, a, b, c),
            Instruction::SetTabUp { a, b, c, k } => {
                let closure = self.frame_closure(frame);
                let object = self.emit(Op::LoadUpval { closure, index: a }, Type::ANY, Rep::Boxed);
                if let Some(node) = self.node_hint(frame, pc, b)
                    && let Some(next) = self.set_node(frame, pc, object, node, (c, k))
                {
                    return next;
                }
                let key = self.constant_value(frame, u32::from(b));
                return self.new_index(frame, pc, object, key, (c, k));
            }
            Instruction::SetField { a, b, c, k } => return self.set_field(frame, pc, a, b, (c, k)),
            Instruction::SetTable { a, b, c, k } => {
                let object = self.read(frame, a);
                let key = self.read(frame, b);
                return self.new_index(frame, pc, object, key, (c, k));
            }
            Instruction::SetI { a, b, c, k } => {
                let object = self.read(frame, a);
                let key = self.constant(Const::Int(i64::from(b)));
                return self.new_index(frame, pc, object, key, (c, k));
            }
            Instruction::AddI { a, b, sc } => return self.arith_immediate(frame, pc, a, b, NumOp::Add, Const::Int(i64::from(sc))),
            Instruction::ShrI { a, b, sc } => return self.arith_immediate(frame, pc, a, b, NumOp::Shr, Const::Int(i64::from(sc))),
            Instruction::ShlI { a, b, sc } => return self.shift_immediate(frame, pc, a, b, sc),
            Instruction::AddK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Add),
            Instruction::SubK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Sub),
            Instruction::MulK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Mul),
            Instruction::ModK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Mod),
            Instruction::PowK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Pow),
            Instruction::DivK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::Div),
            Instruction::IDivK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::IDiv),
            Instruction::BAndK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::BAnd),
            Instruction::BOrK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::BOr),
            Instruction::BXorK { a, b, c } => return self.arith_constant(frame, pc, (a, b, c), NumOp::BXor),
            Instruction::Add { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Add),
            Instruction::Sub { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Sub),
            Instruction::Mul { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Mul),
            Instruction::Mod { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Mod),
            Instruction::Pow { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Pow),
            Instruction::Div { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Div),
            Instruction::IDiv { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::IDiv),
            Instruction::BAnd { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::BAnd),
            Instruction::BOr { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::BOr),
            Instruction::BXor { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::BXor),
            Instruction::Shl { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Shl),
            Instruction::Shr { a, b, c } => return self.arith_registers(frame, pc, (a, b, c), NumOp::Shr),
            Instruction::Unm { a, b } => return self.unary(frame, pc, a, b, NumOp::Neg),
            Instruction::BNot { a, b } => return self.unary(frame, pc, a, b, NumOp::BNot),
            Instruction::Not { a, b } => {
                let value = self.read(frame, b);
                let truthy = self.emit(Op::Unary(UnOp::Truthy, value), Type::BOOL, Rep::Bool);
                let not = self.emit(Op::Unary(UnOp::Not, truthy), Type::BOOL, Rep::Bool);
                self.write(frame, a, not);
            }
            Instruction::Len { a, b } => {
                if !self.speculates(frame, pc) {
                    return self.fallback(frame, pc);
                }
                let value = self.read(frame, b);
                let state = self.state_at(frame, pc);
                let length = self.emit(Op::Len { value, state }, Type::INT, Rep::Int);
                self.write(frame, a, length);
            }
            Instruction::Jmp { .. } => {
                let Some(&target) = frame.flow.successors[pc].first() else {
                    return self.fallback(frame, pc);
                };
                if frame.flow.back_edges[pc] {
                    self.safepoint(frame, target);
                }
                self.jump_to(frame, target);
                return next;
            }
            Instruction::Test { a, k } => {
                let value = self.read(frame, a);
                let truthy = self.emit(Op::Unary(UnOp::Truthy, value), Type::BOOL, Rep::Bool);
                return self.conditional(frame, pc, truthy, k);
            }
            Instruction::TestSet { a, b, k } => return self.test_set(frame, pc, a, b, k),
            Instruction::Eq { a, b, k } => return self.compare_registers(frame, pc, (a, b), Cc::Eq, k),
            Instruction::Lt { a, b, k } => return self.compare_registers(frame, pc, (a, b), Cc::Lt, k),
            Instruction::Le { a, b, k } => return self.compare_registers(frame, pc, (a, b), Cc::Le, k),
            Instruction::EqK { a, b, k } => return self.equal_constant(frame, pc, a, b, k),
            Instruction::EqI { a, sb, k, .. } => {
                let value = self.read(frame, a);
                let equal = self.emit(Op::EqConst(value, Const::Int(i64::from(sb))), Type::BOOL, Rep::Bool);
                return self.conditional(frame, pc, equal, k);
            }
            Instruction::LtI { a, sb, k, .. } => return self.compare_immediate(frame, pc, a, sb, (Cc::Lt, false), k),
            Instruction::LeI { a, sb, k, .. } => return self.compare_immediate(frame, pc, a, sb, (Cc::Le, false), k),
            Instruction::GtI { a, sb, k, .. } => return self.compare_immediate(frame, pc, a, sb, (Cc::Lt, true), k),
            Instruction::GeI { a, sb, k, .. } => return self.compare_immediate(frame, pc, a, sb, (Cc::Le, true), k),
            Instruction::Call { a, b, c } => return self.call(frame, pc, a, b, c),
            Instruction::TailCall { a, b, k, .. } => {
                if frame.site.is_some() {
                    return self.deopt(frame, pc);
                }
                self.flush();
                self.end(Term::TailCall { pc: pc as u32, a, b, k });
                return next;
            }
            Instruction::Return { a, b, k, .. } if b != 0 => return self.ret(frame, pc, a, u32::from(b) - 1, k),
            Instruction::Return0 { .. } => return self.ret(frame, pc, 0, 0, false),
            Instruction::Return1 { a } => return self.ret(frame, pc, a, 1, false),
            Instruction::ForPrep { a, bx } => return self.for_prep(frame, pc, a, bx),
            Instruction::ForLoop { a, bx } => return self.for_loop(frame, pc, a, bx),
            Instruction::TForLoop { a, .. } => {
                let control = self.read(frame, a + 3);
                let ended = self.emit(Op::EqConst(control, Const::Nil), Type::BOOL, Rep::Bool);
                let Some(&target) = frame.flow.successors[pc].first() else {
                    return self.fallback(frame, pc);
                };
                let fall = pc + 1;
                let from = self.block();
                let regs = self.regs.clone();
                let back = self.func.new_block();
                self.end(Term::Branch { cond: ended, then: Edge::to(BlockId(u32::MAX)), other: Edge::to(back) });
                self.edge_to(frame, from, 0, fall);
                self.regs = regs;
                self.current = Some(back);
                self.safepoint(frame, target);
                self.jump_to(frame, target);
                return next;
            }
            Instruction::NewTable { a, b, c, k } => return self.new_table(frame, pc, (a, b, c, k)),
            Instruction::Closure { a, bx } => return self.helper(frame, pc, HelperKind::Fast(sites::FAST_CLOSURE), [u32::from(a), bx, 0]),
            Instruction::SetList { a, b, c, k: false } => return self.helper(frame, pc, HelperKind::Fast(sites::FAST_SET_LIST), [u32::from(a), u32::from(b), c]),
            _ => return self.fallback(frame, pc),
        }
        next
    }

    fn frame_closure(&mut self, frame: &Frame<'_>) -> Closure {
        if frame.chain.is_empty() {
            return Closure::Root;
        }
        let slot = frame.base - 1;
        let value = match self.regs.get(slot as usize).copied() {
            Some(Reg::Val(value, _)) => value,
            _ => self.emit(Op::LoadReg { slot }, Type::ANY, Rep::Boxed),
        };
        let closure = self.emit(Op::Unary(UnOp::Unbox, value), Type::CLOSURE, Rep::Payload);
        Closure::Value(closure)
    }

    fn constant_value(&mut self, frame: &Frame<'_>, index: u32) -> Val {
        let constant = frame.proto.constants.get(index as usize).copied().unwrap_or(LuaValue::Nil);
        let constant = lua_constant(constant);
        self.constant(constant)
    }

    fn load_constant(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, index: u32, next: usize) -> usize {
        if index as usize >= frame.proto.constants.len() {
            return self.fallback(frame, pc);
        }
        let value = self.constant_value(frame, index);
        self.write(frame, a, value);
        next
    }

    fn fallback(&mut self, frame: &mut Frame<'a>, pc: usize) -> usize {
        self.helper(frame, pc, HelperKind::Step, [0; 3])
    }

    fn helper(&mut self, frame: &mut Frame<'a>, pc: usize, kind: HelperKind, args: [u32; 3]) -> usize {
        if frame.site.is_some() {
            return self.deopt(frame, pc);
        }
        self.flush();
        let code = self.emit(Op::Helper(Helper { pc: pc as u32, kind, args }), Type::NONE, Rep::Word);
        self.forget();
        self.switch(frame, pc, code)
    }

    fn switch(&mut self, frame: &mut Frame<'a>, pc: usize, code: Val) -> usize {
        let targets = frame.flow.successors[pc].clone();
        let from = self.block();
        let cases = targets.iter().map(|&target| (target as u32, Edge::to(BlockId(u32::MAX)))).collect();
        self.end(Term::Switch { code, cases });
        for (edge, target) in targets.into_iter().enumerate() {
            self.edge_to(frame, from, edge, target);
        }
        pc + 1
    }

    fn deopt(&mut self, frame: &Frame<'_>, pc: usize) -> usize {
        let state = self.state_at(frame, pc);
        self.end(Term::Deopt { state });
        pc + 1
    }

    fn safepoint(&mut self, frame: &Frame<'_>, target: usize) {
        if frame.site.is_some() {
            return;
        }
        let state = self.state_at(frame, target);
        self.effect(Op::Safepoint { pc: target as u32, state });
    }

    fn conditional(&mut self, frame: &mut Frame<'a>, pc: usize, cond: Val, k: bool) -> usize {
        let Some(target) = flow::test_target(frame.proto, pc) else {
            return self.fallback(frame, pc);
        };
        let (then, other) = if k { (target, pc + 2) } else { (pc + 2, target) };
        self.branch_to(frame, cond, then, other);
        pc + 2
    }

    fn test_set(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, k: bool) -> usize {
        let Some(target) = flow::test_target(frame.proto, pc) else {
            return self.fallback(frame, pc);
        };
        let value = self.read(frame, b);
        let truthy = self.emit(Op::Unary(UnOp::Truthy, value), Type::BOOL, Rep::Bool);
        let from = self.block();
        let regs = self.regs.clone();
        let copy = self.func.new_block();
        let (then, other) = if k { (Edge::to(copy), Edge::to(BlockId(u32::MAX))) } else { (Edge::to(BlockId(u32::MAX)), Edge::to(copy)) };
        self.end(Term::Branch { cond: truthy, then, other });
        self.edge_to(frame, from, usize::from(k), pc + 2);
        self.regs = regs;
        self.current = Some(copy);
        self.write(frame, a, value);
        self.jump_to(frame, target);
        pc + 2
    }

    fn compare_registers(&mut self, frame: &mut Frame<'a>, pc: usize, (a, b): (u8, u8), cc: Cc, k: bool) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let left = self.read(frame, a);
        let right = self.read(frame, b);
        let state = self.state_at(frame, pc);
        let cond = self.emit(Op::Compare { cc, left, right, state }, Type::BOOL, Rep::Bool);
        self.conditional(frame, pc, cond, k)
    }

    fn compare_immediate(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, sb: i32, (cc, reversed): (Cc, bool), k: bool) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let register = self.read(frame, a);
        let immediate = self.constant(Const::Int(i64::from(sb)));
        let (left, right) = if reversed { (immediate, register) } else { (register, immediate) };
        let state = self.state_at(frame, pc);
        let cond = self.emit(Op::Compare { cc, left, right, state }, Type::BOOL, Rep::Bool);
        self.conditional(frame, pc, cond, k)
    }

    fn equal_constant(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, k: bool) -> usize {
        let constant = frame.proto.constants.get(usize::from(b)).copied().unwrap_or(LuaValue::Nil);
        let comparable = match constant {
            LuaValue::String(_) => (self.facts.interned)(constant),
            LuaValue::Nil | LuaValue::Boolean(_) | LuaValue::Integer(_) | LuaValue::Float(_) => true,
            _ => false,
        };
        if !comparable {
            return self.fallback(frame, pc);
        }
        let value = self.read(frame, a);
        let equal = self.emit(Op::EqConst(value, lua_constant(constant)), Type::BOOL, Rep::Bool);
        self.conditional(frame, pc, equal, k)
    }

    fn arith(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, op: NumOp, (left, right): (Val, Val)) -> usize {
        let state = self.state_at(frame, pc);
        let value = self.emit(Op::Arith { op, left, right, state }, Type::NUM, Rep::Boxed);
        self.write(frame, a, value);
        pc + 2
    }

    fn arith_registers(&mut self, frame: &mut Frame<'a>, pc: usize, (a, b, c): (u8, u8, u8), op: NumOp) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let left = self.read(frame, b);
        let right = self.read(frame, c);
        self.arith(frame, pc, a, op, (left, right))
    }

    fn arith_constant(&mut self, frame: &mut Frame<'a>, pc: usize, (a, b, c): (u8, u8, u8), op: NumOp) -> usize {
        let numeric = matches!(frame.proto.constants.get(usize::from(c)), Some(LuaValue::Integer(_) | LuaValue::Float(_)));
        if !numeric || !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let left = self.read(frame, b);
        let right = self.constant_value(frame, u32::from(c));
        self.arith(frame, pc, a, op, (left, right))
    }

    fn arith_immediate(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, op: NumOp, immediate: Const) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let left = self.read(frame, b);
        let right = self.constant(immediate);
        self.arith(frame, pc, a, op, (left, right))
    }

    fn shift_immediate(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, sc: i32) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let left = self.constant(Const::Int(i64::from(sc)));
        let right = self.read(frame, b);
        self.arith(frame, pc, a, NumOp::Shl, (left, right))
    }

    fn unary(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, op: NumOp) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let operand = self.read(frame, b);
        let state = self.state_at(frame, pc);
        let value = self.emit(Op::Arith { op, left: operand, right: operand, state }, Type::NUM, Rep::Boxed);
        self.write(frame, a, value);
        pc + 1
    }

    fn table_of(&mut self, value: Val, state: StateId) -> Val {
        if self.func.ty(value) == Type::TABLE {
            return value;
        }
        self.emit(Op::GuardType { value, ty: Type::TABLE, state }, Type::TABLE, Rep::Boxed)
    }

    fn node_hint(&self, frame: &Frame<'_>, pc: usize, constant: u8) -> Option<(NodeHint, u64)> {
        if !self.facts.inline_caches || !frame.chain.is_empty() {
            return None;
        }
        let hint = (*self.facts.nodes.get(pc)?)?;
        let key = frame.proto.constants.get(usize::from(constant)).copied()?;
        let LuaValue::String(string) = key else {
            return None;
        };
        (self.facts.interned)(key).then_some((hint, string.packed()))
    }

    fn get_node(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, object: Val, (hint, key): (NodeHint, u64)) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let state = self.state_at(frame, pc);
        let table = self.table_of(object, state);
        self.effect(Op::GuardNode { table, node: hint.node, nodes: hint.nodes, key, state });
        let value = self.emit(Op::LoadNode { table, node: hint.node }, Type::ANY, Rep::Boxed);
        let value = self.emit(Op::GuardNotNil { value, state }, Type::ANY.without(Type::NIL), Rep::Boxed);
        self.write(frame, a, value);
        pc + 1
    }

    /// Storing nil removes the key, which the generic path does.
    fn set_node(&mut self, frame: &mut Frame<'a>, pc: usize, object: Val, (hint, key): (NodeHint, u64), value: (u8, bool)) -> Option<usize> {
        if !self.speculates(frame, pc) {
            return None;
        }
        let stored = self.stored(frame, value);
        if !self.func.ty(stored).intersect(Type::NIL).is_none() {
            return None;
        }
        let state = self.state_at(frame, pc);
        let table = self.table_of(object, state);
        self.effect(Op::GuardNode { table, node: hint.node, nodes: hint.nodes, key, state });
        self.effect(Op::StoreNode { table, node: hint.node, value: stored, state });
        Some(pc + 1)
    }

    fn get_field(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, object: Val, c: u8) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let state = self.state_at(frame, pc);
        let value = self.field_value(frame, pc, object, c, state);
        self.write(frame, a, value);
        pc + 1
    }

    /// Follows Maglev's `TryBuildNamedAccess` and `TryBuildPolymorphicPropertyAccess`.
    fn field_value(&mut self, frame: &Frame<'_>, pc: usize, object: Val, c: u8, state: StateId) -> Val {
        let methods = self.method_feedback(frame, pc);
        if let [method] = methods.as_slice() {
            let table = self.table_of(object, state);
            return self.method_chain(table, method, state);
        }
        let entries = self.field_feedback(frame, pc);
        match entries.as_slice() {
            [] => {
                let table = self.table_of(object, state);
                let key = self.constant_value(frame, u32::from(c));
                self.emit(Op::Index { object: table, key, state }, Type::ANY, Rep::Boxed)
            }
            [entry] => {
                let table = self.table_of(object, state);
                self.effect(Op::GuardShape { table, shape: entry.shape.packed(), state });
                self.slot_value(table, entry.slot, state)
            }
            _ => {
                let table = self.table_of(object, state);
                self.polymorphic_load(table, &entries, state)
            }
        }
    }

    fn slot_value(&mut self, table: Val, slot: u32, state: StateId) -> Val {
        if slot == ABSENT {
            self.effect(Op::GuardNoMeta { table, bit: event_bit(Event::Index), state });
            return self.constant(Const::Nil);
        }
        let value = self.emit(Op::LoadSlot { table, slot }, Type::ANY, Rep::Boxed);
        self.emit(Op::GuardNotNil { value, state }, Type::ANY.without(Type::NIL), Rep::Boxed)
    }

    fn polymorphic_load(&mut self, table: Val, entries: &[Entry], state: StateId) -> Val {
        let join = self.func.new_block();
        let result = self.func.add_param(join, Type::ANY, Rep::Boxed);
        for entry in entries {
            let cond = self.emit(Op::ShapeIs { table, shape: entry.shape.packed() }, Type::BOOL, Rep::Bool);
            let hit = self.func.new_block();
            let miss = self.func.new_block();
            self.end(Term::Branch { cond, then: Edge::to(hit), other: Edge::to(miss) });
            self.current = Some(hit);
            let value = self.slot_value(table, entry.slot, state);
            self.end(Term::Jump(Edge { block: join, args: vec![value] }));
            self.current = Some(miss);
        }
        self.end(Term::Deopt { state });
        self.current = Some(join);
        result
    }

    /// The class reads do not depend on the receiver, so the hoisting pass can move them out of loops.
    fn method_chain(&mut self, table: Val, method: &MethodEntry, state: StateId) -> Val {
        self.effect(Op::GuardShape { table, shape: method.receiver.packed(), state });
        let metatable = self.emit(Op::GuardHasMeta { table, state }, Type::TABLE, Rep::Payload);
        self.effect(Op::GuardShape { table: metatable, shape: method.metatable_shape.packed(), state });
        let index = self.emit(Op::LoadSlot { table: metatable, slot: method.index_slot }, Type::ANY, Rep::Boxed);
        let index = self.emit(Op::GuardType { value: index, ty: Type::TABLE, state }, Type::TABLE, Rep::Boxed);
        self.effect(Op::GuardEq { value: index, expected: method.class.packed(), state });
        let class = self.constant(Const::Gc { tag: 5, payload: method.class.packed() });
        self.effect(Op::GuardShape { table: class, shape: method.class_shape.packed(), state });
        let value = self.emit(Op::LoadSlot { table: class, slot: method.class_slot }, Type::ANY, Rep::Boxed);
        self.emit(Op::GuardNotNil { value, state }, Type::ANY.without(Type::NIL), Rep::Boxed)
    }

    fn field_feedback(&self, frame: &Frame<'_>, pc: usize) -> Vec<Entry> {
        if !self.facts.inline_caches {
            return Vec::new();
        }
        sites::field_entries(frame.proto, pc)
    }

    fn method_feedback(&self, frame: &Frame<'_>, pc: usize) -> Vec<MethodEntry> {
        if !self.facts.inline_caches {
            return Vec::new();
        }
        sites::method_entries(frame.proto, pc)
    }

    fn self_op(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, c: u8) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let object = self.read(frame, b);
        let state = self.state_at(frame, pc);
        let method = self.field_value(frame, pc, object, c, state);
        self.write(frame, a + 1, object);
        self.write(frame, a, method);
        pc + 1
    }

    fn index(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, object: Val, key: Val) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let state = self.state_at(frame, pc);
        let table = self.table_of(object, state);
        let value = self.emit(Op::Index { object: table, key, state }, Type::ANY, Rep::Boxed);
        self.write(frame, a, value);
        pc + 1
    }

    fn stored(&mut self, frame: &Frame<'_>, (c, k): (u8, bool)) -> Val {
        if k {
            return self.constant_value(frame, u32::from(c));
        }
        self.read(frame, c)
    }

    fn new_index(&mut self, frame: &mut Frame<'a>, pc: usize, object: Val, key: Val, value: (u8, bool)) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let value = self.stored(frame, value);
        let state = self.state_at(frame, pc);
        let table = self.table_of(object, state);
        self.effect(Op::NewIndex { object: table, key, value, state });
        pc + 1
    }

    /// A store that adds a key goes through the helper, which moves the table to its next shape.
    fn set_field(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, value: (u8, bool)) -> usize {
        if !self.speculates(frame, pc) {
            return self.fallback(frame, pc);
        }
        let object = self.read(frame, a);
        let entries: Vec<Entry> = self.field_feedback(frame, pc).into_iter().filter(|entry| entry.slot != ABSENT).collect();
        let adds = self.facts.inline_caches && !sites::add_entries(frame.proto, pc).is_empty();
        let stored = self.stored(frame, value);
        let state = self.state_at(frame, pc);
        match entries.as_slice() {
            [entry] if !adds => {
                let table = self.table_of(object, state);
                self.effect(Op::GuardShape { table, shape: entry.shape.packed(), state });
                self.effect(Op::StoreSlot { table, slot: entry.slot, value: stored, state });
            }
            _ => {
                let table = self.table_of(object, state);
                let key = self.constant_value(frame, u32::from(b));
                self.effect(Op::NewIndex { object: table, key, value: stored, state });
            }
        }
        pc + 1
    }

    fn new_table(&mut self, frame: &mut Frame<'a>, pc: usize, (a, b, c, k): (u8, u8, u32, bool)) -> usize {
        let extra = match (k, frame.proto.code.get(pc + 1)) {
            (false, Some(Instruction::ExtraArg { .. })) => 0,
            (true, Some(Instruction::ExtraArg { ax })) => u64::from(*ax),
            _ => return self.fallback(frame, pc),
        };
        let array = u64::from(c) + extra * 1024;
        if array > i32::MAX as u64 {
            return self.fallback(frame, pc);
        }
        self.helper(frame, pc, HelperKind::Fast(sites::FAST_NEW_TABLE), [u32::from(a), array as u32, u32::from(b)])
    }

    fn call(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, b: u8, c: u8) -> usize {
        if self.speculates(frame, pc)
            && let Some(candidate) = self.inline_candidate(frame, pc, (a, b, c))
        {
            return self.inline_call(frame, pc, (a, b, c), candidate);
        }
        if frame.site.is_some() {
            return self.deopt(frame, pc);
        }
        self.flush();
        let code = self.emit(Op::CallLua { pc: pc as u32, a, b, c }, Type::NONE, Rep::Word);
        self.forget();
        self.switch(frame, pc, code)
    }

    fn inline_candidate(&self, frame: &Frame<'_>, pc: usize, (a, b, c): (u8, u8, u8)) -> Option<Candidate<'a>> {
        if !self.facts.inline_caches || b == 0 || c == 0 || frame.chain.len() >= self.facts.policy.depth {
            return None;
        }
        let Some(InlineCache::Callee(callee)) = frame.proto.exec.feedback.get(pc) else {
            return None;
        };
        let proto = callee.proto?;
        let function = (self.facts.proto_of)(proto)?;
        let top = Self::slot(frame, a) as usize + 1 + usize::from(function.max_stack);
        if top > crate::vm::REGISTER_SPAN {
            return None;
        }
        inline::candidate(function, proto, &self.facts.policy)
    }

    /// Follows Maglev's `BuildInlineFunction`.
    fn inline_call(&mut self, frame: &mut Frame<'a>, pc: usize, (a, b, c): (u8, u8, u8), candidate: Candidate<'a>) -> usize {
        let state = self.state_at(frame, pc);
        let function = self.read(frame, a);
        let closure = self.emit(Op::GuardType { value: function, ty: Type::CLOSURE, state }, Type::CLOSURE, Rep::Boxed);
        let proto = self.emit(Op::LoadProto { closure }, Type::NONE, Rep::Word);
        self.effect(Op::GuardEq { value: proto, expected: candidate.reference.packed(), state });
        let base = Self::slot(frame, a) + 1;
        let arguments = u32::from(b) - 1;
        let nil = self.constant(Const::Nil);
        for parameter in arguments..u32::from(candidate.proto.num_params) {
            self.ensure_slot(base + parameter);
            self.regs[(base + parameter) as usize] = Reg::Val(nil, true);
        }
        self.write(frame, a, closure);
        let wanted = i32::from(c) - 1;
        let mut chain = frame.chain.clone();
        chain.push(InlineFrame { base, caller_pc: pc as u32 + 1, wanted, proto: candidate.reference.packed() });
        let site = frame.site.unwrap_or(pc as u32);
        let mut callee = self.frame(candidate.proto, base, chain, Some(site), Some((Self::slot(frame, a), wanted)));
        for slot in base + u32::from(candidate.proto.num_params).max(arguments)..base + u32::from(candidate.proto.max_stack) {
            self.ensure_slot(slot);
            self.regs[slot as usize] = Reg::Mem;
        }
        self.jump_to(&mut callee, 0);
        self.run(&mut callee);
        self.finish_inline(frame, pc, callee.returns)
    }

    fn finish_inline(&mut self, frame: &mut Frame<'a>, pc: usize, returns: Vec<Incoming>) -> usize {
        if returns.is_empty() {
            self.current = None;
            return pc + 1;
        }
        let block = self.merge(frame, pc + 1, returns);
        self.current = Some(block);
        pc + 1
    }

    fn ret(&mut self, frame: &mut Frame<'a>, pc: usize, first: u8, count: u32, close: bool) -> usize {
        let Some((function_slot, wanted)) = frame.caller else {
            self.flush();
            self.end(Term::Return { pc: pc as u32, first, count, close });
            return pc + 1;
        };
        if close {
            return self.deopt(frame, pc);
        }
        let results: Vec<Val> = (0..count).map(|index| self.read(frame, first + index as u8)).collect();
        let nil = self.constant(Const::Nil);
        let wanted = wanted.max(0) as u32;
        for index in 0..wanted {
            let value = results.get(index as usize).copied().unwrap_or(nil);
            let slot = function_slot + index;
            self.ensure_slot(slot);
            self.regs[slot as usize] = Reg::Val(value, true);
        }
        for slot in function_slot + wanted..self.regs.len() as u32 {
            self.regs[slot as usize] = Reg::Mem;
        }
        let from = self.block();
        self.end(Term::Jump(Edge::to(BlockId(u32::MAX))));
        frame.returns.push(Incoming { from, edge: 0, regs: self.regs.clone() });
        pc + 1
    }

    /// Follows `forprep` in lvm.c. The count, step and start feed the bounds-check pass.
    fn for_prep(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, bx: u32) -> usize {
        if frame.site.is_some() || !self.speculates(frame, pc) {
            return self.helper(frame, pc, HelperKind::Fast(sites::FAST_FOR_PREP), [u32::from(a), bx, 0]);
        }
        let values: Vec<Val> = (a..a + 3).map(|register| self.read(frame, register)).collect();
        if values.iter().any(|&value| self.entry_type(frame, value).intersect(Type::INT).is_none()) {
            return self.helper(frame, pc, HelperKind::Fast(sites::FAST_FOR_PREP), [u32::from(a), bx, 0]);
        }
        let state = self.state_at(frame, pc);
        let operands: Vec<Val> = values.into_iter().map(|value| self.integer(value, state)).collect();
        let (start, limit, step) = (operands[0], operands[1], operands[2]);
        let step_constant = match self.func.op(step) {
            Op::Const(Const::Int(step)) => Some(*step),
            _ => None,
        };
        if step_constant.is_none() {
            let zero = self.constant(Const::Int(0));
            let is_zero = self.emit(Op::Binary(BinOp::ICmp(Cc::Eq), step, zero), Type::BOOL, Rep::Bool);
            self.effect(Op::GuardFalse { cond: is_zero, state });
        }
        let skip = self.skip_test(start, limit, step, step_constant);
        let count = self.emit(Op::Pure(PureFn::LoopCount, Box::new([start, limit, step])), Type::INT, Rep::Int);
        let body = pc + 1;
        let exit = pc + bx as usize + 2;
        let from = self.block();
        let regs = self.regs.clone();
        let enter = self.func.new_block();
        self.end(Term::Branch { cond: skip, then: Edge::to(BlockId(u32::MAX)), other: Edge::to(enter) });
        self.edge_to(frame, from, 0, exit);
        self.regs = regs;
        self.current = Some(enter);
        self.write(frame, a, count);
        self.write(frame, a + 1, step);
        self.write(frame, a + 2, start);
        frame.counted[body] = step_constant;
        frame.inline_prep[body] = true;
        self.jump_to(frame, body);
        pc + 1
    }

    fn skip_test(&mut self, start: Val, limit: Val, step: Val, step_constant: Option<i64>) -> Val {
        let up = self.emit(Op::Binary(BinOp::ICmp(Cc::Lt), limit, start), Type::BOOL, Rep::Bool);
        let down = self.emit(Op::Binary(BinOp::ICmp(Cc::Lt), start, limit), Type::BOOL, Rep::Bool);
        match step_constant {
            Some(step) if step > 0 => up,
            Some(_) => down,
            None => {
                let zero = self.constant(Const::Int(0));
                let positive = self.emit(Op::Binary(BinOp::ICmp(Cc::Lt), zero, step), Type::BOOL, Rep::Bool);
                self.emit(Op::Select { cond: positive, yes: up, no: down }, Type::BOOL, Rep::Bool)
            }
        }
    }

    fn integer(&mut self, value: Val, state: StateId) -> Val {
        if self.func.ty(value) == Type::INT {
            return value;
        }
        self.emit(Op::GuardType { value, ty: Type::INT, state }, Type::INT, Rep::Boxed)
    }

    fn for_loop(&mut self, frame: &mut Frame<'a>, pc: usize, a: u8, bx: u32) -> usize {
        let target = (pc + 1).saturating_sub(bx as usize);
        if frame.site.is_some() || !self.speculates(frame, pc) || !frame.inline_prep[target] {
            return self.fallback_loop(frame, pc, target);
        }
        let state = self.state_at(frame, pc);
        let count = self.read(frame, a);
        let count = self.integer(count, state);
        let step = self.read(frame, a + 1);
        let step = self.integer(step, state);
        let index = self.read(frame, a + 2);
        let index = self.integer(index, state);
        let zero = self.constant(Const::Int(0));
        let done = self.emit(Op::Binary(BinOp::ICmp(Cc::Eq), count, zero), Type::BOOL, Rep::Bool);
        let from = self.block();
        let regs = self.regs.clone();
        let again = self.func.new_block();
        self.end(Term::Branch { cond: done, then: Edge::to(BlockId(u32::MAX)), other: Edge::to(again) });
        self.edge_to(frame, from, 0, pc + 1);
        self.regs = regs;
        self.current = Some(again);
        let one = self.constant(Const::Int(1));
        let remaining = self.emit(Op::Binary(BinOp::ISub, count, one), Type::INT, Rep::Int);
        let advanced = self.emit(Op::Binary(BinOp::IAdd, index, step), Type::INT, Rep::Int);
        self.write(frame, a, remaining);
        self.write(frame, a + 2, advanced);
        self.safepoint(frame, target);
        self.jump_to(frame, target);
        pc + 1
    }

    /// The helper runs the loop instruction and answers the pc it continues at; the back edge still polls.
    fn fallback_loop(&mut self, frame: &mut Frame<'a>, pc: usize, target: usize) -> usize {
        if frame.site.is_some() {
            return self.deopt(frame, pc);
        }
        self.flush();
        let code = self.emit(Op::Helper(Helper { pc: pc as u32, kind: HelperKind::Step, args: [0; 3] }), Type::NONE, Rep::Word);
        self.forget();
        let from = self.block();
        let back = self.func.new_block();
        self.end(Term::Switch { code, cases: vec![(target as u32, Edge::to(back)), ((pc + 1) as u32, Edge::to(BlockId(u32::MAX)))] });
        self.edge_to(frame, from, 1, pc + 1);
        self.current = Some(back);
        self.safepoint(frame, target);
        self.jump_to(frame, target);
        pc + 1
    }
}

fn event_bit(event: Event) -> u8 {
    event.absence_bit().unwrap_or(0)
}

pub(crate) fn lua_constant(value: LuaValue) -> Const {
    match value {
        LuaValue::Nil => Const::Nil,
        LuaValue::Boolean(flag) => Const::Bool(flag),
        LuaValue::Integer(integer) => Const::Int(integer),
        LuaValue::Float(float) => Const::Float(float),
        LuaValue::String(string) => Const::Gc { tag: 4, payload: string.packed() },
        LuaValue::Table(table) => Const::Gc { tag: 5, payload: table.packed() },
        LuaValue::Closure(closure) => Const::Gc { tag: 6, payload: closure.packed() },
        LuaValue::Native(native) => Const::Gc { tag: 7, payload: native.packed() },
        LuaValue::Thread(thread) => Const::Gc { tag: 8, payload: thread.packed() },
        LuaValue::Userdata(userdata) => Const::Gc { tag: 9, payload: userdata.packed() },
        LuaValue::LightUserdata(address) => Const::Gc { tag: 10, payload: address as u64 },
    }
}
