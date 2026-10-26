#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct Val(pub(crate) u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct BlockId(pub(crate) u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct StateId(pub(crate) u32);

/// A set of Lua value kinds; bit `n` is the kind whose `Value` tag is `n`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Type(u16);

impl Type {
    pub(crate) const NONE: Type = Type(0);
    pub(crate) const NIL: Type = Type(1 << 0);
    pub(crate) const BOOL: Type = Type(1 << 1);
    pub(crate) const INT: Type = Type(1 << 2);
    pub(crate) const FLOAT: Type = Type(1 << 3);
    pub(crate) const TABLE: Type = Type(1 << 5);
    pub(crate) const CLOSURE: Type = Type(1 << 6);
    pub(crate) const NUM: Type = Type(Self::INT.0 | Self::FLOAT.0);
    pub(crate) const ANY: Type = Type((1 << 11) - 1);

    pub(crate) fn of_tag(tag: u8) -> Type {
        Type(1 << tag)
    }

    pub(crate) fn union(self, other: Type) -> Type {
        Type(self.0 | other.0)
    }

    pub(crate) fn intersect(self, other: Type) -> Type {
        Type(self.0 & other.0)
    }

    pub(crate) fn without(self, other: Type) -> Type {
        Type(self.0 & !other.0)
    }

    pub(crate) fn within(self, other: Type) -> bool {
        self.0 & !other.0 == 0
    }

    pub(crate) fn is_none(self) -> bool {
        self.0 == 0
    }

    pub(crate) fn tag(self) -> Option<u8> {
        (self.0.count_ones() == 1).then(|| self.0.trailing_zeros() as u8)
    }

    pub(crate) fn may_be_falsy(self) -> bool {
        !self.intersect(Self::NIL.union(Self::BOOL)).is_none()
    }

    pub(crate) fn unboxed(self) -> Rep {
        match self.tag() {
            Some(2) => Rep::Int,
            Some(3) => Rep::Float,
            Some(1) => Rep::Bool,
            Some(tag) if tag >= 4 => Rep::Payload,
            _ => Rep::Boxed,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Rep {
    None,
    Boxed,
    Int,
    Float,
    Bool,
    Payload,
    Word,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Const {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Gc { tag: u8, payload: u64 },
}

impl Const {
    pub(crate) fn ty(self) -> Type {
        match self {
            Const::Nil => Type::NIL,
            Const::Bool(_) => Type::BOOL,
            Const::Int(_) => Type::INT,
            Const::Float(_) => Type::FLOAT,
            Const::Gc { tag, .. } => Type::of_tag(tag),
        }
    }

    pub(crate) fn tag(self) -> u8 {
        match self {
            Const::Nil => 0,
            Const::Bool(_) => 1,
            Const::Int(_) => 2,
            Const::Float(_) => 3,
            Const::Gc { tag, .. } => tag,
        }
    }

    pub(crate) fn payload(self) -> u64 {
        match self {
            Const::Nil => 0,
            Const::Bool(flag) => u64::from(flag),
            Const::Int(integer) => integer as u64,
            Const::Float(float) => float.to_bits(),
            Const::Gc { payload, .. } => payload,
        }
    }

    pub(crate) fn same(self, other: Const) -> bool {
        self.tag() == other.tag() && self.payload() == other.payload()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Cc {
    Eq,
    Lt,
    Le,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum UnOp {
    INeg,
    IBnot,
    FNeg,
    IToF,
    Truthy,
    Not,
    Box,
    Unbox,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum BinOp {
    IAdd,
    ISub,
    IMul,
    IMod,
    IDiv,
    IBand,
    IBor,
    IBxor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FIDiv,
    ICmp(Cc),
    FCmp(Cc),
    BoolEq,
}

/// The same Rust functions the interpreter runs, so results match.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum PureFn {
    FMod,
    FPow,
    Shl,
    Shr,
    LoopCount,
    LtIF,
    LeIF,
    LtFI,
    LeFI,
    EqFI,
}

/// Builtins compiled code calls directly: none allocates, runs Lua code or raises.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum BuiltinFn {
    Type,
    RawEqual,
    RawGet,
    GetMetatable,
    SetMetatable,
}

impl BuiltinFn {
    pub(crate) fn is_pure(self) -> bool {
        matches!(self, BuiltinFn::Type | BuiltinFn::RawEqual)
    }

    pub(crate) fn writes(self) -> bool {
        self == BuiltinFn::SetMetatable
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum NumOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    IDiv,
    Pow,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
    Neg,
    BNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Closure {
    Root,
    Value(Val),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum HelperKind {
    Step,
    Fast(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Helper {
    pub(crate) pc: u32,
    pub(crate) kind: HelperKind,
    pub(crate) args: [u32; 3],
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Op {
    Param,
    Const(Const),
    LoadReg { slot: u32 },
    StoreReg { slot: u32, value: Val },
    LoadUpval { closure: Closure, index: u8 },
    StoreUpval { closure: Closure, index: u8, value: Val },
    Unary(UnOp, Val),
    Binary(BinOp, Val, Val),
    Pure(PureFn, Box<[Val]>),
    EqConst(Val, Const),
    Select { cond: Val, yes: Val, no: Val },
    Arith { op: NumOp, left: Val, right: Val, state: StateId },
    Compare { cc: Cc, left: Val, right: Val, state: StateId },
    Len { value: Val, state: StateId },
    ShapeIs { table: Val, shape: u64 },
    LoadSlot { table: Val, slot: u32 },
    LoadProto { closure: Val },
    LoadArray { table: Val, index: Val, state: StateId },
    GuardType { value: Val, ty: Type, state: StateId },
    GuardShape { table: Val, shape: u64, state: StateId },
    GuardHasMeta { table: Val, state: StateId },
    GuardNoMeta { table: Val, bit: u8, state: StateId },
    GuardEq { value: Val, expected: u64, state: StateId },
    GuardFalse { cond: Val, state: StateId },
    GuardNotNil { value: Val, state: StateId },
    GuardRange { table: Val, low: Val, high: Val, state: StateId },
    Index { object: Val, key: Val, state: StateId },
    NewIndex { object: Val, key: Val, value: Val, state: StateId },
    StoreSlot { table: Val, slot: u32, value: Val, state: StateId },
    AddSlot { table: Val, key: Val, shape: u64, next: u64, slot: u32, value: Val, state: StateId },
    /// Follows LuaJIT's `HREFK`: the hash node at `node` of a table with `nodes` nodes holds the interned string `key`.
    GuardNode { table: Val, node: u32, nodes: u32, key: u64, state: StateId },
    LoadNode { table: Val, node: u32 },
    StoreNode { table: Val, node: u32, value: Val, state: StateId },
    Helper(Helper),
    Builtin { function: BuiltinFn, args: Box<[Val]>, state: StateId },
    /// The collector step scans only the registers below absolute slot `top`.
    NewTable { hash_log: u8, top: u32, shape: Option<u64>, fields: Box<[Val]>, state: StateId, after: StateId },
    Materialize { object: Box<Virtual> },
    GuardAbsent { table: Val, bit: u8, state: StateId },
    CallLua { pc: u32, a: u8, b: u8, c: u8 },
    /// `weight` is the loop's length in bytecodes, the work one iteration stands for.
    Safepoint { pc: u32, weight: u32, state: StateId },
}

impl Op {
    pub(crate) fn operands(&self) -> Vec<Val> {
        let mut values = Vec::new();
        self.for_each_operand(|value| values.push(value));
        values
    }

    pub(crate) fn for_each_operand(&self, mut f: impl FnMut(Val)) {
        match self {
            Op::Param | Op::Const(_) | Op::LoadReg { .. } | Op::Helper(_) | Op::CallLua { .. } | Op::Safepoint { .. } => {}
            Op::NewTable { fields, .. } => fields.iter().for_each(|&field| f(field)),
            Op::Materialize { object } => object.for_each_operand(&mut f),
            Op::StoreReg { value, .. } | Op::Unary(_, value) | Op::EqConst(value, _) | Op::Len { value, .. } => f(*value),
            Op::LoadUpval { closure, .. } => each_closure(*closure, &mut f),
            Op::StoreUpval { closure, value, .. } => {
                each_closure(*closure, &mut f);
                f(*value);
            }
            Op::Binary(_, left, right) | Op::Arith { left, right, .. } | Op::Compare { left, right, .. } => {
                f(*left);
                f(*right);
            }
            Op::Pure(_, args) | Op::Builtin { args, .. } => args.iter().for_each(|&arg| f(arg)),
            Op::LoadSlot { table, .. } | Op::LoadNode { table, .. } | Op::ShapeIs { table, .. } => f(*table),
            Op::Select { cond, yes, no } => {
                f(*cond);
                f(*yes);
                f(*no);
            }
            Op::LoadProto { closure } => f(*closure),
            Op::LoadArray { table, index, .. } => {
                f(*table);
                f(*index);
            }
            Op::GuardType { value, .. } | Op::GuardEq { value, .. } | Op::GuardNotNil { value, .. } => f(*value),
            Op::GuardShape { table, .. } | Op::GuardNode { table, .. } | Op::GuardHasMeta { table, .. } | Op::GuardNoMeta { table, .. } | Op::GuardAbsent { table, .. } => f(*table),
            Op::GuardFalse { cond, .. } => f(*cond),
            Op::GuardRange { table, low, high, .. } => {
                f(*table);
                f(*low);
                f(*high);
            }
            Op::Index { object, key, .. } => {
                f(*object);
                f(*key);
            }
            Op::NewIndex { object, key, value, .. } => {
                f(*object);
                f(*key);
                f(*value);
            }
            Op::StoreSlot { table, value, .. } | Op::StoreNode { table, value, .. } => {
                f(*table);
                f(*value);
            }
            Op::AddSlot { table, key, value, .. } => {
                f(*table);
                f(*key);
                f(*value);
            }
        }
    }

    pub(crate) fn map_operands(&mut self, mut f: impl FnMut(Val) -> Val) {
        match self {
            Op::Param | Op::Const(_) | Op::LoadReg { .. } | Op::Helper(_) | Op::CallLua { .. } | Op::Safepoint { .. } => {}
            Op::NewTable { fields, .. } => fields.iter_mut().for_each(|field| *field = f(*field)),
            Op::Materialize { object } => object.map_operands(&mut f),
            Op::StoreReg { value, .. } | Op::Unary(_, value) | Op::EqConst(value, _) => *value = f(*value),
            Op::LoadUpval { closure, .. } => map_closure(closure, &mut f),
            Op::StoreUpval { closure, value, .. } => {
                map_closure(closure, &mut f);
                *value = f(*value);
            }
            Op::Binary(_, left, right) => {
                *left = f(*left);
                *right = f(*right);
            }
            Op::Pure(_, args) | Op::Builtin { args, .. } => args.iter_mut().for_each(|arg| *arg = f(*arg)),
            Op::LoadSlot { table, .. } | Op::LoadNode { table, .. } | Op::ShapeIs { table, .. } => *table = f(*table),
            Op::Arith { left, right, .. } | Op::Compare { left, right, .. } => {
                *left = f(*left);
                *right = f(*right);
            }
            Op::Len { value, .. } => *value = f(*value),
            Op::Select { cond, yes, no } => {
                *cond = f(*cond);
                *yes = f(*yes);
                *no = f(*no);
            }
            Op::LoadProto { closure } => *closure = f(*closure),
            Op::LoadArray { table, index, .. } => {
                *table = f(*table);
                *index = f(*index);
            }
            Op::GuardType { value, .. } | Op::GuardEq { value, .. } | Op::GuardNotNil { value, .. } => *value = f(*value),
            Op::GuardShape { table, .. } | Op::GuardNode { table, .. } | Op::GuardHasMeta { table, .. } | Op::GuardNoMeta { table, .. } | Op::GuardAbsent { table, .. } => *table = f(*table),
            Op::GuardFalse { cond, .. } => *cond = f(*cond),
            Op::GuardRange { table, low, high, .. } => {
                *table = f(*table);
                *low = f(*low);
                *high = f(*high);
            }
            Op::Index { object, key, .. } => {
                *object = f(*object);
                *key = f(*key);
            }
            Op::NewIndex { object, key, value, .. } => {
                *object = f(*object);
                *key = f(*key);
                *value = f(*value);
            }
            Op::StoreSlot { table, value, .. } | Op::StoreNode { table, value, .. } => {
                *table = f(*table);
                *value = f(*value);
            }
            Op::AddSlot { table, key, value, .. } => {
                *table = f(*table);
                *key = f(*key);
                *value = f(*value);
            }
        }
    }

    pub(crate) fn state(&self) -> Option<StateId> {
        match self {
            Op::GuardType { state, .. }
            | Op::GuardShape { state, .. }
            | Op::GuardHasMeta { state, .. }
            | Op::GuardNoMeta { state, .. }
            | Op::GuardEq { state, .. }
            | Op::GuardFalse { state, .. }
            | Op::GuardNotNil { state, .. }
            | Op::GuardRange { state, .. }
            | Op::Index { state, .. }
            | Op::NewIndex { state, .. }
            | Op::StoreSlot { state, .. }
            | Op::AddSlot { state, .. }
            | Op::GuardNode { state, .. }
            | Op::StoreNode { state, .. }
            | Op::Safepoint { state, .. }
            | Op::Arith { state, .. }
            | Op::Compare { state, .. }
            | Op::Len { state, .. }
            | Op::Builtin { state, .. }
            | Op::NewTable { state, .. }
            | Op::GuardAbsent { state, .. }
            | Op::LoadArray { state, .. } => Some(*state),
            _ => None,
        }
    }

    pub(crate) fn set_state(&mut self, new: StateId) {
        match self {
            Op::GuardType { state, .. }
            | Op::GuardShape { state, .. }
            | Op::GuardHasMeta { state, .. }
            | Op::GuardNoMeta { state, .. }
            | Op::GuardEq { state, .. }
            | Op::GuardFalse { state, .. }
            | Op::GuardNotNil { state, .. }
            | Op::GuardRange { state, .. }
            | Op::Index { state, .. }
            | Op::NewIndex { state, .. }
            | Op::StoreSlot { state, .. }
            | Op::AddSlot { state, .. }
            | Op::GuardNode { state, .. }
            | Op::StoreNode { state, .. }
            | Op::Safepoint { state, .. }
            | Op::Arith { state, .. }
            | Op::Compare { state, .. }
            | Op::Len { state, .. }
            | Op::Builtin { state, .. }
            | Op::NewTable { state, .. }
            | Op::GuardAbsent { state, .. }
            | Op::LoadArray { state, .. } => *state = new,
            _ => {}
        }
    }

    pub(crate) fn is_pure(&self) -> bool {
        match self {
            Op::Builtin { function, .. } => function.is_pure(),
            _ => matches!(self, Op::Const(_) | Op::Unary(..) | Op::Binary(..) | Op::Pure(..) | Op::EqConst(..) | Op::Select { .. }),
        }
    }

    pub(crate) fn is_removable(&self) -> bool {
        match self {
            Op::Builtin { function, .. } => !function.writes(),
            _ => self.is_pure() || matches!(self, Op::Param | Op::LoadReg { .. } | Op::LoadSlot { .. } | Op::LoadNode { .. } | Op::LoadProto { .. } | Op::LoadUpval { .. } | Op::ShapeIs { .. }),
        }
    }
}

fn each_closure(closure: Closure, f: &mut impl FnMut(Val)) {
    if let Closure::Value(value) = closure {
        f(value);
    }
}

fn map_closure(closure: &mut Closure, f: &mut impl FnMut(Val) -> Val) {
    if let Closure::Value(value) = closure {
        *value = f(*value);
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Edge {
    pub(crate) block: BlockId,
    pub(crate) args: Vec<Val>,
}

impl Edge {
    pub(crate) fn to(block: BlockId) -> Self {
        Self { block, args: Vec::new() }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Term {
    Open,
    Jump(Edge),
    Branch { cond: Val, then: Edge, other: Edge },
    Switch { code: Val, cases: Vec<(u32, Edge)> },
    Return { pc: u32, first: u8, count: u32, close: bool },
    TailCall { pc: u32, a: u8, b: u8, k: bool },
    Deopt { state: StateId },
}

impl Term {
    pub(crate) fn edges(&self) -> Vec<&Edge> {
        match self {
            Term::Jump(edge) => vec![edge],
            Term::Branch { then, other, .. } => vec![then, other],
            Term::Switch { cases, .. } => cases.iter().map(|(_, edge)| edge).collect(),
            Term::Open | Term::Return { .. } | Term::TailCall { .. } | Term::Deopt { .. } => Vec::new(),
        }
    }

    pub(crate) fn edges_mut(&mut self) -> Vec<&mut Edge> {
        match self {
            Term::Jump(edge) => vec![edge],
            Term::Branch { then, other, .. } => vec![then, other],
            Term::Switch { cases, .. } => cases.iter_mut().map(|(_, edge)| edge).collect(),
            Term::Open | Term::Return { .. } | Term::TailCall { .. } | Term::Deopt { .. } => Vec::new(),
        }
    }

    pub(crate) fn successors(&self) -> Vec<BlockId> {
        self.edges().into_iter().map(|edge| edge.block).collect()
    }

    pub(crate) fn map_operands(&mut self, mut f: impl FnMut(Val) -> Val) {
        match self {
            Term::Branch { cond, .. } => *cond = f(*cond),
            Term::Switch { code, .. } => *code = f(*code),
            _ => {}
        }
        for edge in self.edges_mut() {
            edge.args.iter_mut().for_each(|arg| *arg = f(*arg));
        }
    }

    pub(crate) fn operands(&self) -> Vec<Val> {
        let mut values = Vec::new();
        let mut copy = self.clone();
        copy.map_operands(|value| {
            values.push(value);
            value
        });
        values
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct InlineFrame {
    pub(crate) base: u32,
    pub(crate) caller_pc: u32,
    pub(crate) wanted: i32,
    pub(crate) proto: u64,
    /// Reached by a tail call, which the interpreter marks on the frame it reuses.
    pub(crate) tail: bool,
}

/// A table the code does not allocate, following V8's captured objects and LuaJIT's `snap_unsink` (lj_snap.c).
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Virtual {
    pub(crate) value: Val,
    pub(crate) hash_log: u8,
    pub(crate) shape: u64,
    pub(crate) fields: Vec<Val>,
    pub(crate) meta: Option<Val>,
    pub(crate) original: Option<(Val, Val)>,
}

impl Virtual {
    pub(crate) fn for_each_operand(&self, f: &mut impl FnMut(Val)) {
        self.fields.iter().for_each(|&field| f(field));
        if let Some(meta) = self.meta {
            f(meta);
        }
        if let Some((flag, original)) = self.original {
            f(flag);
            f(original);
        }
    }

    pub(crate) fn map_operands(&mut self, f: &mut impl FnMut(Val) -> Val) {
        self.fields.iter_mut().for_each(|field| *field = f(*field));
        if let Some(meta) = &mut self.meta {
            *meta = f(*meta);
        }
        if let Some((flag, original)) = &mut self.original {
            *flag = f(*flag);
            *original = f(*original);
        }
    }
}

/// Follows V8 Maglev's eager deopt frame (`GetDeoptFrameForEagerDeopt`, src/maglev/maglev-graph-builder.cc).
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct FrameState {
    pub(crate) pc: u32,
    pub(crate) frames: Vec<InlineFrame>,
    pub(crate) slots: Vec<(u32, Val)>,
    pub(crate) site: u32,
    /// Follows V8's lazy deoptimization: the instruction ran code the frame cannot account for, and no speculation failed.
    pub(crate) lazy: bool,
    pub(crate) objects: Vec<Virtual>,
}

impl FrameState {
    pub(crate) fn object(&self, value: Val) -> Option<&Virtual> {
        self.objects.iter().find(|object| object.value == value)
    }

    pub(crate) fn for_each_operand(&self, mut f: impl FnMut(Val)) {
        for &(_, value) in &self.slots {
            match self.object(value) {
                Some(object) => object.for_each_operand(&mut f),
                None => f(value),
            }
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Inst {
    pub(crate) op: Op,
    pub(crate) ty: Type,
    pub(crate) rep: Rep,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Block {
    pub(crate) params: Vec<Val>,
    pub(crate) insts: Vec<Val>,
    pub(crate) term: Term,
    pub(crate) cold: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Counted {
    pub(crate) index: Val,
    pub(crate) count: Val,
    pub(crate) step: i64,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Loop {
    pub(crate) header: BlockId,
    pub(crate) preheader: BlockId,
    pub(crate) blocks: Vec<BlockId>,
    pub(crate) entry_state: StateId,
    pub(crate) counted: Option<Counted>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct Func {
    pub(crate) insts: Vec<Inst>,
    pub(crate) blocks: Vec<Block>,
    pub(crate) states: Vec<FrameState>,
    pub(crate) entry: Option<BlockId>,
    pub(crate) osr: Vec<(u32, BlockId)>,
    pub(crate) loops: Vec<Loop>,
    pub(crate) float_guess: bool,
    pub(crate) avoid: Vec<u32>,
    pub(crate) observed: Vec<(u32, u8)>,
    pub(crate) type_names: Option<[u64; 8]>,
    /// Shapes the code moves tables into; the prototype keeps them alive.
    pub(crate) shapes: Vec<crate::arena::Ref<crate::shape::Shape>>,
    /// Allocations may be removed: no memory limit makes an allocation able to fail.
    pub(crate) sink_allocations: bool,
}

pub(crate) const KIND_INT: u8 = 1;
pub(crate) const KIND_FLOAT: u8 = 2;
pub(crate) const KIND_OTHER: u8 = 4;

impl Func {
    pub(crate) fn new_block(&mut self) -> BlockId {
        self.blocks.push(Block {
            params: Vec::new(),
            insts: Vec::new(),
            term: Term::Open,
            cold: false,
        });
        BlockId(self.blocks.len() as u32 - 1)
    }

    pub(crate) fn block(&self, block: BlockId) -> &Block {
        &self.blocks[block.0 as usize]
    }

    pub(crate) fn block_mut(&mut self, block: BlockId) -> &mut Block {
        &mut self.blocks[block.0 as usize]
    }

    pub(crate) fn inst(&self, value: Val) -> &Inst {
        &self.insts[value.0 as usize]
    }

    pub(crate) fn inst_mut(&mut self, value: Val) -> &mut Inst {
        &mut self.insts[value.0 as usize]
    }

    pub(crate) fn op(&self, value: Val) -> &Op {
        &self.inst(value).op
    }

    pub(crate) fn ty(&self, value: Val) -> Type {
        self.inst(value).ty
    }

    pub(crate) fn rep(&self, value: Val) -> Rep {
        self.inst(value).rep
    }

    pub(crate) fn add_inst(&mut self, op: Op, ty: Type, rep: Rep) -> Val {
        self.insts.push(Inst { op, ty, rep });
        Val(self.insts.len() as u32 - 1)
    }

    pub(crate) fn append(&mut self, block: BlockId, op: Op, ty: Type, rep: Rep) -> Val {
        let value = self.add_inst(op, ty, rep);
        self.block_mut(block).insts.push(value);
        value
    }

    pub(crate) fn add_param(&mut self, block: BlockId, ty: Type, rep: Rep) -> Val {
        let value = self.add_inst(Op::Param, ty, rep);
        self.block_mut(block).params.push(value);
        value
    }

    pub(crate) fn add_state(&mut self, state: FrameState) -> StateId {
        if self.states.last() == Some(&state) {
            return StateId(self.states.len() as u32 - 1);
        }
        self.states.push(state);
        StateId(self.states.len() as u32 - 1)
    }

    pub(crate) fn state(&self, state: StateId) -> &FrameState {
        &self.states[state.0 as usize]
    }


    pub(crate) fn reverse_postorder(&self) -> Vec<BlockId> {
        let mut seen = vec![false; self.blocks.len()];
        let mut order = Vec::new();
        let roots: Vec<BlockId> = self.entry.into_iter().chain(self.osr.iter().map(|&(_, block)| block)).collect();
        for root in roots {
            self.postorder_from(root, &mut seen, &mut order);
        }
        order.reverse();
        order
    }

    fn postorder_from(&self, root: BlockId, seen: &mut [bool], order: &mut Vec<BlockId>) {
        if seen[root.0 as usize] {
            return;
        }
        seen[root.0 as usize] = true;
        let mut stack = vec![(root, 0usize)];
        while let Some((block, next)) = stack.pop() {
            let successors = self.block(block).term.successors();
            let Some(&successor) = successors.get(next) else {
                order.push(block);
                continue;
            };
            stack.push((block, next + 1));
            if !seen[successor.0 as usize] {
                seen[successor.0 as usize] = true;
                stack.push((successor, 0));
            }
        }
    }

    pub(crate) fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let mut predecessors = vec![Vec::new(); self.blocks.len()];
        for block in self.reverse_postorder() {
            for successor in self.block(block).term.successors() {
                predecessors[successor.0 as usize].push(block);
            }
        }
        predecessors
    }

    pub(crate) fn definitions(&self) -> Vec<Option<BlockId>> {
        let mut defined = vec![None; self.insts.len()];
        for block in self.reverse_postorder() {
            let data = self.block(block);
            for &value in data.params.iter().chain(&data.insts) {
                defined[value.0 as usize] = Some(block);
            }
        }
        defined
    }


    pub(crate) fn rewrite_uses(&mut self, map: &[Val]) {
        let resolve = |value: Val| resolve(map, value);
        for block in &mut self.blocks {
            block.term.map_operands(resolve);
        }
        for inst in &mut self.insts {
            inst.op.map_operands(resolve);
        }
        for state in &mut self.states {
            let placeholders: Vec<Val> = state.objects.iter().map(|object| object.value).collect();
            state.slots.iter_mut().filter(|(_, value)| !placeholders.contains(value)).for_each(|(_, value)| *value = resolve(*value));
            let mut map = resolve;
            state.objects.iter_mut().for_each(|object| object.map_operands(&mut map));
        }
    }

    pub(crate) fn identity_map(&self) -> Vec<Val> {
        (0..self.insts.len() as u32).map(Val).collect()
    }

    /// Cooper, Harvey and Kennedy's iterative algorithm; all entries hang off one virtual root.
    pub(crate) fn dominators(&self) -> Vec<Option<BlockId>> {
        let order = self.reverse_postorder();
        let mut position = vec![usize::MAX; self.blocks.len()];
        for (index, block) in order.iter().enumerate() {
            position[block.0 as usize] = index + 1;
        }
        let predecessors = self.predecessors();
        let roots: Vec<BlockId> = self.entry.into_iter().chain(self.osr.iter().map(|&(_, block)| block)).collect();
        let mut idom: Vec<Option<BlockId>> = vec![None; self.blocks.len()];
        for &root in &roots {
            idom[root.0 as usize] = Some(VIRTUAL_ROOT);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &order {
                if roots.contains(&block) {
                    continue;
                }
                let mut new = None;
                for &predecessor in &predecessors[block.0 as usize] {
                    if idom[predecessor.0 as usize].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => predecessor,
                        Some(current) => intersect(&idom, &position, predecessor, current),
                    });
                }
                if new.is_some() && idom[block.0 as usize] != new {
                    idom[block.0 as usize] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    pub(crate) fn dominates(idom: &[Option<BlockId>], dominator: BlockId, block: BlockId) -> bool {
        let mut current = block;
        loop {
            if current == dominator {
                return true;
            }
            match idom.get(current.0 as usize).copied().flatten() {
                Some(parent) if parent != VIRTUAL_ROOT => current = parent,
                _ => return false,
            }
        }
    }
}

const VIRTUAL_ROOT: BlockId = BlockId(u32::MAX);

fn intersect(idom: &[Option<BlockId>], position: &[usize], mut left: BlockId, mut right: BlockId) -> BlockId {
    let at = |block: BlockId| if block == VIRTUAL_ROOT { 0 } else { position[block.0 as usize] };
    let up = |block: BlockId| if block == VIRTUAL_ROOT { VIRTUAL_ROOT } else { idom[block.0 as usize].unwrap_or(VIRTUAL_ROOT) };
    while left != right {
        while at(left) > at(right) {
            left = up(left);
        }
        while at(right) > at(left) {
            right = up(right);
        }
    }
    left
}

pub(crate) fn resolve(map: &[Val], mut value: Val) -> Val {
    while let Some(&next) = map.get(value.0 as usize) {
        if next == value {
            break;
        }
        value = next;
    }
    value
}
