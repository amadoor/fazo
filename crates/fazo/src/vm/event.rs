#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Index,
    NewIndex,
    Len,
    Eq,
    Lt,
    Le,
    Concat,
    Call,
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IDiv,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
    Unm,
    BNot,
    Close,
    ToString,
    Pairs,
    Gc,
}

pub(crate) const MODE_ABSENCE_BIT: u8 = 1 << 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Field {
    Metatable,
    Name,
    Mode,
}

impl Field {
    pub(crate) const ALL: [Field; 3] = [Field::Metatable, Field::Name, Field::Mode];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Field::Metatable => "__metatable",
            Field::Name => "__name",
            Field::Mode => "__mode",
        }
    }

    pub(crate) fn index(self) -> usize {
        Event::ALL.len() + self as usize
    }
}

impl Event {
    pub(crate) const ALL: [Event; 26] = [
        Event::Index,
        Event::NewIndex,
        Event::Len,
        Event::Eq,
        Event::Lt,
        Event::Le,
        Event::Concat,
        Event::Call,
        Event::Add,
        Event::Sub,
        Event::Mul,
        Event::Mod,
        Event::Pow,
        Event::Div,
        Event::IDiv,
        Event::BAnd,
        Event::BOr,
        Event::BXor,
        Event::Shl,
        Event::Shr,
        Event::Unm,
        Event::BNot,
        Event::Close,
        Event::ToString,
        Event::Pairs,
        Event::Gc,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Event::Index => "__index",
            Event::NewIndex => "__newindex",
            Event::Len => "__len",
            Event::Eq => "__eq",
            Event::Lt => "__lt",
            Event::Le => "__le",
            Event::Concat => "__concat",
            Event::Call => "__call",
            Event::Add => "__add",
            Event::Sub => "__sub",
            Event::Mul => "__mul",
            Event::Mod => "__mod",
            Event::Pow => "__pow",
            Event::Div => "__div",
            Event::IDiv => "__idiv",
            Event::BAnd => "__band",
            Event::BOr => "__bor",
            Event::BXor => "__bxor",
            Event::Shl => "__shl",
            Event::Shr => "__shr",
            Event::Unm => "__unm",
            Event::BNot => "__bnot",
            Event::Close => "__close",
            Event::ToString => "__tostring",
            Event::Pairs => "__pairs",
            Event::Gc => "__gc",
        }
    }

    /// Only events PUC caches with `fasttm` (ltm.h) have a bit; `__mode` has its own.
    pub(crate) fn absence_bit(self) -> Option<u8> {
        let index = match self {
            Event::Index => 0,
            Event::NewIndex => 1,
            Event::Len => 2,
            Event::Eq => 3,
            Event::Lt => 4,
            Event::Le => 5,
            Event::Gc => 6,
            _ => return None,
        };
        Some(1 << index)
    }
}
