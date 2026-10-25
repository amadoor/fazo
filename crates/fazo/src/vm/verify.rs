use super::ops::arithmetic_target;
use crate::{
    Vm,
    arena::Ref,
    instruction::Instruction,
    object::Frame,
    proto::{Proto, UpvalueDesc},
};

const MAX_NESTING: usize = 1000;

impl Vm {
    pub(crate) fn verify_proto(&self, proto: Ref<Proto>) -> Result<(), &'static str> {
        check(&self.shape_of(proto, 0)?)
    }

    pub(crate) fn verify_image(&self) -> Result<(), &'static str> {
        for (proto, _) in self.heap.protos.iter() {
            self.verify_proto(proto)?;
        }
        for (_, thread) in self.heap.threads.iter() {
            for frame in &thread.frames {
                let Frame::Lua(frame) = frame else {
                    continue;
                };
                let closure = self.heap.closures.try_get(frame.closure).ok_or("a frame's closure is missing")?;
                let function = self.heap.protos.try_get(closure.proto).ok_or("a frame's prototype is missing")?;
                if frame.pc() >= function.code.len() {
                    return Err("a frame's program counter is out of range");
                }
            }
        }
        Ok(())
    }

    fn shape_of(&self, proto: Ref<Proto>, depth: usize) -> Result<Shape<'_>, &'static str> {
        if depth > MAX_NESTING {
            return Err("functions nest too deeply");
        }
        let function = self.heap.protos.try_get(proto).ok_or("a prototype is missing")?;
        let children = function
            .protos
            .iter()
            .map(|child| self.shape_of(*child, depth + 1))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Shape {
            code: &function.code,
            num_params: function.num_params,
            max_stack: function.max_stack,
            constant_count: function.constants.len(),
            upvalues: &function.upvalues,
            children,
        })
    }
}

pub(super) struct Shape<'a> {
    pub(super) code: &'a [Instruction],
    pub(super) num_params: u8,
    pub(super) max_stack: u8,
    pub(super) constant_count: usize,
    pub(super) upvalues: &'a [UpvalueDesc],
    pub(super) children: Vec<Shape<'a>>,
}

pub(super) fn check(shape: &Shape<'_>) -> Result<(), &'static str> {
    let Some(&last) = shape.code.last() else {
        return Err("empty function");
    };
    if !is_terminal(last) {
        return Err("function does not end with a return or jump");
    }
    if shape.num_params > shape.max_stack {
        return Err("a function's parameters do not fit its stack");
    }
    let needs_close = shape
        .children
        .iter()
        .any(|child| child.upvalues.iter().any(|description| description.in_stack))
        || shape
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Tbc { .. } | Instruction::TForPrep { .. }));
    for (pc, &instruction) in shape.code.iter().enumerate() {
        check_extra_argument(shape.code, pc, instruction)?;
        check_registers(shape, instruction)?;
        check_references(shape, shape.code, pc, instruction)?;
        check_successors(shape.code, pc, instruction)?;
        check_close(needs_close, instruction)?;
        check_placement(shape.code, pc, instruction)?;
    }
    shape.children.iter().try_for_each(check)
}

fn check_close(needs_close: bool, instruction: Instruction) -> Result<(), &'static str> {
    let closes = match instruction {
        Instruction::Return { k, .. } => k,
        Instruction::Return0 { .. } | Instruction::Return1 { .. } => false,
        _ => return Ok(()),
    };
    if needs_close && !closes {
        return Err("a return skips the function's pending closes");
    }
    Ok(())
}

/// `EXTRAARG` is data for the instruction before it, so it must never run or be a jump target.
fn check_extra_argument(code: &[Instruction], pc: usize, instruction: Instruction) -> Result<(), &'static str> {
    let owner = match instruction {
        Instruction::LoadKx { .. } | Instruction::NewTable { .. } => true,
        Instruction::SetList { k, .. } => k,
        _ => false,
    };
    if owner && !matches!(code.get(pc + 1), Some(Instruction::ExtraArg { .. })) {
        return Err("an instruction that needs an extra argument lacks it");
    }
    if let Instruction::ExtraArg { .. } = instruction {
        let follows_owner = pc > 0
            && match code[pc - 1] {
                Instruction::LoadKx { .. } | Instruction::NewTable { .. } => true,
                Instruction::SetList { k, .. } => k,
                _ => false,
            };
        if !follows_owner {
            return Err("an extra argument does not follow the instruction it belongs to");
        }
    }
    Ok(())
}

/// Metamethod instructions run only after their arithmetic instruction, and `VARARGPREP` only first.
fn check_placement(code: &[Instruction], pc: usize, instruction: Instruction) -> Result<(), &'static str> {
    match instruction {
        Instruction::MmBin { .. } | Instruction::MmBinI { .. } | Instruction::MmBinK { .. } => {
            match pc.checked_sub(1).map(|previous| code[previous]) {
                Some(previous) if arithmetic_target(previous).is_some() => Ok(()),
                _ => Err("a metamethod instruction does not follow an arithmetic instruction"),
            }
        }
        Instruction::VarargPrep { .. } if pc != 0 => Err("VARARGPREP is not the first instruction"),
        _ => Ok(()),
    }
}

fn is_metamethod(instruction: Instruction) -> bool {
    matches!(instruction, Instruction::MmBin { .. } | Instruction::MmBinI { .. } | Instruction::MmBinK { .. })
}

fn is_terminal(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Jmp { .. } | Instruction::Return { .. } | Instruction::Return0 { .. } | Instruction::Return1 { .. }
    )
}

fn check_registers(shape: &Shape<'_>, instruction: Instruction) -> Result<(), &'static str> {
    match highest_register(instruction) {
        Some(register) if register >= usize::from(shape.max_stack) => Err("register out of range"),
        _ => Ok(()),
    }
}

fn highest_register(instruction: Instruction) -> Option<usize> {
    use Instruction::*;
    let at = |register: u8| Some(usize::from(register));
    let highest = |registers: &[Option<usize>]| registers.iter().flatten().copied().max();
    let value_operand = |c: u8, k: bool| if k { None } else { at(c) };
    match instruction {
        LoadI { a, .. }
        | LoadF { a, .. }
        | LoadK { a, .. }
        | LoadKx { a }
        | LoadFalse { a }
        | LFalseSkip { a }
        | LoadTrue { a }
        | SetUpval { a, .. }
        | GetUpval { a, .. }
        | GetTabUp { a, .. }
        | NewTable { a, .. }
        | Close { a }
        | Tbc { a }
        | ErrNil { a, .. }
        | Closure { a, .. }
        | Return1 { a }
        | EqI { a, .. }
        | LtI { a, .. }
        | LeI { a, .. }
        | GtI { a, .. }
        | GeI { a, .. }
        | Test { a, .. }
        | MmBinI { a, .. }
        | MmBinK { a, .. }
        | EqK { a, .. } => at(a),
        ForLoop { a, .. } | ForPrep { a, .. } => Some(usize::from(a) + 2),
        TForPrep { a, .. } | TForLoop { a, .. } => Some(usize::from(a) + 3),
        Move { a, b }
        | GetI { a, b, .. }
        | GetField { a, b, .. }
        | Unm { a, b }
        | BNot { a, b }
        | Not { a, b }
        | Len { a, b }
        | AddI { a, b, .. }
        | AddK { a, b, .. }
        | SubK { a, b, .. }
        | MulK { a, b, .. }
        | ModK { a, b, .. }
        | PowK { a, b, .. }
        | DivK { a, b, .. }
        | IDivK { a, b, .. }
        | BAndK { a, b, .. }
        | BOrK { a, b, .. }
        | BXorK { a, b, .. }
        | ShrI { a, b, .. }
        | ShlI { a, b, .. }
        | MmBin { a, b, .. }
        | Eq { a, b, .. }
        | Lt { a, b, .. }
        | Le { a, b, .. } => highest(&[at(a), at(b)]),
        LoadNil { a, b } => Some(usize::from(a) + usize::from(b)),
        Concat { a, b } => Some(usize::from(a) + usize::from(b).max(1) - 1),
        GetTable { a, b, c }
        | GetVarg { a, b, c }
        | Add { a, b, c }
        | Sub { a, b, c }
        | Mul { a, b, c }
        | Mod { a, b, c }
        | Pow { a, b, c }
        | Div { a, b, c }
        | IDiv { a, b, c }
        | BAnd { a, b, c }
        | BOr { a, b, c }
        | BXor { a, b, c }
        | Shl { a, b, c }
        | Shr { a, b, c } => highest(&[at(a), at(b), at(c)]),
        SetTabUp { c, k, .. } => value_operand(c, k),
        SetTable { a, b, c, k } => highest(&[at(a), at(b), value_operand(c, k)]),
        SetI { a, c, k, .. } | SetField { a, c, k, .. } => highest(&[at(a), value_operand(c, k)]),
        SelfOp { a, b, .. } => highest(&[Some(usize::from(a) + 1), at(a), at(b)]),
        TestSet { b, .. } => at(b),
        Call { a, b, c } | TailCall { a, b, c, .. } => highest(&[
            at(a),
            Some(usize::from(a) + usize::from(b).max(1) - 1),
            Some(usize::from(a) + usize::from(c).max(2) - 2),
        ]),
        Return { a, b, .. } => match b {
            1 => None,
            0 => at(a),
            _ => Some(usize::from(a) + usize::from(b) - 2),
        },
        TForCall { a, c } => highest(&[Some(usize::from(a) + 5), Some(usize::from(a) + 2 + usize::from(c))]),
        SetList { a, b, .. } => Some(usize::from(a) + usize::from(b)),
        Vararg { a, b, c, k } => highest(&[at(a), Some(usize::from(a) + usize::from(c).max(2) - 2), if k { at(b) } else { None }]),
        Return0 { .. } | VarargPrep { .. } | Jmp { .. } | ExtraArg { .. } => None,
    }
}

fn check_references(shape: &Shape<'_>, code: &[Instruction], pc: usize, instruction: Instruction) -> Result<(), &'static str> {
    use Instruction::*;
    match instruction {
        GetUpval { b, .. } | SetUpval { b, .. } => upvalue(shape, b),
        GetTabUp { b, c, .. } => {
            upvalue(shape, b)?;
            constant(shape, u32::from(c))
        }
        SetTabUp { a, b, c, k } => {
            upvalue(shape, a)?;
            constant(shape, u32::from(b))?;
            rk(shape, c, k)
        }
        SetTable { c, k, .. } | SetI { c, k, .. } => rk(shape, c, k),
        SetField { b, c, k, .. } => {
            constant(shape, u32::from(b))?;
            rk(shape, c, k)
        }
        LoadK { bx, .. } => constant(shape, bx),
        LoadKx { .. } => match code.get(pc + 1) {
            Some(ExtraArg { ax }) => constant(shape, *ax),
            _ => Ok(()),
        },
        GetField { c, .. } | SelfOp { c, .. } => constant(shape, u32::from(c)),
        EqK { b, .. } | MmBinK { b, .. } => constant(shape, u32::from(b)),
        AddK { c, .. } | SubK { c, .. } | MulK { c, .. } | ModK { c, .. } | PowK { c, .. } | DivK { c, .. } | IDivK { c, .. }
        | BAndK { c, .. } | BOrK { c, .. } | BXorK { c, .. } => constant(shape, u32::from(c)),
        ErrNil { bx, .. } if bx > 0 => constant(shape, bx - 1),
        Closure { bx, .. } => closure(shape, bx),
        _ => Ok(()),
    }
}

fn upvalue(shape: &Shape<'_>, index: u8) -> Result<(), &'static str> {
    if usize::from(index) < shape.upvalues.len() {
        Ok(())
    } else {
        Err("upvalue index out of range")
    }
}

fn constant(shape: &Shape<'_>, index: u32) -> Result<(), &'static str> {
    if (index as usize) < shape.constant_count {
        Ok(())
    } else {
        Err("constant index out of range")
    }
}

fn rk(shape: &Shape<'_>, operand: u8, k: bool) -> Result<(), &'static str> {
    if k { constant(shape, u32::from(operand)) } else { Ok(()) }
}

fn closure(shape: &Shape<'_>, index: u32) -> Result<(), &'static str> {
    let Some(child) = shape.children.get(index as usize) else {
        return Err("prototype index out of range");
    };
    for description in child.upvalues {
        let limit = if description.in_stack {
            usize::from(shape.max_stack)
        } else {
            shape.upvalues.len()
        };
        if usize::from(description.index) >= limit {
            return Err("closure upvalue index out of range");
        }
    }
    Ok(())
}

fn check_successors(code: &[Instruction], pc: usize, instruction: Instruction) -> Result<(), &'static str> {
    use Instruction::*;
    let next = pc + 1;
    if instruction.is_test() || matches!(instruction, LFalseSkip { .. }) {
        in_code(code, pc as i64 + 2)?;
    }
    if instruction.is_test() && !matches!(code[next], Jmp { .. }) {
        return Err("test is not followed by a jump");
    }
    match instruction {
        Jmp { sj } => in_code(code, next as i64 + i64::from(sj)),
        ForLoop { bx, .. } | TForLoop { bx, .. } => in_code(code, next as i64 - i64::from(bx)),
        ForPrep { bx, .. } => in_code(code, pc as i64 + 2 + i64::from(bx)),
        TForPrep { bx, .. } => {
            in_code(code, next as i64 + i64::from(bx))?;
            match code.get(next + bx as usize) {
                Some(TForCall { .. }) => Ok(()),
                _ => Err("a generic loop's prepare does not reach its call"),
            }
        }
        TForCall { .. } => match code.get(next) {
            Some(TForLoop { .. }) => Ok(()),
            _ => Err("a generic loop's call is not followed by its loop"),
        },
        _ => Ok(()),
    }
}

fn in_code(code: &[Instruction], target: i64) -> Result<(), &'static str> {
    if 0 <= target && target < code.len() as i64 {
        let landing = code[target as usize];
        if let Instruction::ExtraArg { .. } = landing {
            return Err("a jump lands on an extra argument");
        }
        if is_metamethod(landing) {
            return Err("a jump lands on a metamethod instruction");
        }
        Ok(())
    } else {
        Err("jump target out of range")
    }
}
