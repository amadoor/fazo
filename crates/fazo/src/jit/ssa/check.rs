use super::ir::{BinOp, BlockId, Func, Op, Rep, StateId, Term, UnOp, Val};

pub(crate) fn check(func: &Func, reps_selected: bool) -> Result<(), String> {
    for (index, block) in func.blocks.iter().enumerate() {
        if let Some(edge) = block.term.edges().into_iter().find(|edge| edge.block.0 as usize >= func.blocks.len()) {
            return Err(format!("block{index} jumps to a block that was never built ({:?})", edge.block));
        }
    }
    let idom = func.dominators();
    let defined = func.definitions();
    let mut position = vec![usize::MAX; func.insts.len()];
    for block in func.reverse_postorder() {
        let data = func.block(block);
        for (index, &value) in data.insts.iter().enumerate() {
            position[value.0 as usize] = index + 1;
        }
        for &param in &data.params {
            position[param.0 as usize] = 0;
        }
    }
    let context = Context { func, idom: &idom, defined: &defined, position: &position };
    for block in func.reverse_postorder() {
        context.check_block(block, reps_selected)?;
    }
    Ok(())
}

struct Context<'a> {
    func: &'a Func,
    idom: &'a [Option<BlockId>],
    defined: &'a [Option<BlockId>],
    position: &'a [usize],
}

impl Context<'_> {
    fn check_block(&self, block: BlockId, reps_selected: bool) -> Result<(), String> {
        let data = self.func.block(block);
        for (index, &value) in data.insts.iter().enumerate() {
            let op = &self.func.inst(value).op;
            for operand in op.operands() {
                self.dominated(operand, block, index + 1, value)?;
            }
            if let Some(state) = op.state() {
                self.check_state(state, block, index + 1, value)?;
            }
            if reps_selected {
                self.check_reps(value)?;
            }
        }
        self.check_term(block, &data.term, data.insts.len() + 1)
    }

    fn check_term(&self, block: BlockId, term: &Term, at: usize) -> Result<(), String> {
        if matches!(term, Term::Open) {
            return Err(format!("block{} has no terminator", block.0));
        }
        if let Term::Deopt { state } = term {
            self.check_state(*state, block, at, Val(u32::MAX))?;
        }
        for operand in term.operands() {
            self.dominated(operand, block, at, Val(u32::MAX))?;
        }
        for edge in term.edges() {
            let params = self.func.block(edge.block).params.len();
            if params != edge.args.len() {
                return Err(format!("block{} passes {} arguments to block{}, which takes {params}", block.0, edge.args.len(), edge.block.0));
            }
        }
        Ok(())
    }

    fn check_state(&self, state: StateId, block: BlockId, at: usize, user: Val) -> Result<(), String> {
        let Some(data) = self.func.states.get(state.0 as usize) else {
            return Err(format!("v{} names missing state s{}", user.0, state.0));
        };
        let mut failure = Ok(());
        data.for_each_operand(|value| {
            if failure.is_ok() {
                failure = self.dominated(value, block, at, user);
            }
        });
        failure?;
        Ok(())
    }

    fn dominated(&self, operand: Val, block: BlockId, at: usize, user: Val) -> Result<(), String> {
        let Some(definition) = self.defined.get(operand.0 as usize).copied().flatten() else {
            return Err(format!("v{} uses v{}, which no reachable block defines", user.0, operand.0));
        };
        if definition == block {
            if self.position[operand.0 as usize] < at {
                return Ok(());
            }
            return Err(format!("v{} uses v{} before its definition in block{}", user.0, operand.0, block.0));
        }
        if Func::dominates(self.idom, definition, block) {
            return Ok(());
        }
        Err(format!("v{} in block{} uses v{} from block{}, which does not dominate it", user.0, block.0, operand.0, definition.0))
    }

    fn check_reps(&self, value: Val) -> Result<(), String> {
        let inst = self.func.inst(value);
        let expect = |operand: Val, rep: Rep| -> Result<(), String> {
            let actual = self.func.rep(operand);
            if actual == rep {
                return Ok(());
            }
            Err(format!("v{} reads v{} as {rep}, but it is {actual}", value.0, operand.0))
        };
        match &inst.op {
            Op::Binary(op, left, right) => {
                let rep = binary_operand_rep(*op);
                expect(*left, rep)?;
                expect(*right, rep)
            }
            Op::Unary(op, operand) => match unary_operand_rep(*op) {
                Some(rep) => expect(*operand, rep),
                None => Ok(()),
            },
            Op::StoreReg { value: stored, .. } | Op::StoreSlot { value: stored, .. } | Op::AddSlot { value: stored, .. } | Op::StoreNode { value: stored, .. } | Op::NewIndex { value: stored, .. } => expect(*stored, Rep::Boxed),
            Op::LoadSlot { table, .. } | Op::LoadNode { table, .. } | Op::GuardNode { table, .. } | Op::ShapeIs { table, .. } | Op::GuardShape { table, .. } | Op::GuardHasMeta { table, .. } | Op::GuardNoMeta { table, .. } | Op::GuardAbsent { table, .. } => {
                expect(*table, Rep::Payload)
            }
            Op::GuardFalse { cond, .. } => expect(*cond, Rep::Bool),
            _ => Ok(()),
        }
    }
}

pub(crate) fn binary_operand_rep(op: BinOp) -> Rep {
    match op {
        BinOp::IAdd | BinOp::ISub | BinOp::IMul | BinOp::IMod | BinOp::IDiv | BinOp::IBand | BinOp::IBor | BinOp::IBxor | BinOp::ICmp(_) => Rep::Int,
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FIDiv | BinOp::FCmp(_) => Rep::Float,
        BinOp::BoolEq => Rep::Bool,
    }
}

pub(crate) fn unary_operand_rep(op: UnOp) -> Option<Rep> {
    match op {
        UnOp::INeg | UnOp::IBnot | UnOp::IToF => Some(Rep::Int),
        UnOp::FNeg | UnOp::FSqrt => Some(Rep::Float),
        UnOp::NumToF => Some(Rep::Boxed),
        UnOp::Not => Some(Rep::Bool),
        UnOp::Truthy | UnOp::Box | UnOp::Unbox => None,
    }
}
