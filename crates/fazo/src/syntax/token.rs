use crate::{arena::Ref, object::LuaString};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    Goto,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,
    IntDiv,
    Concat,
    Dots,
    EqEq,
    GreaterEq,
    LessEq,
    NotEq,
    Shl,
    Shr,
    DbColon,
    Eos,
    Float,
    Int,
    Name,
    Str,
    Char(u8),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Value {
    None,
    Int(i64),
    Float(f64),
    Str(Ref<LuaString>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Token {
    pub kind: Kind,
    pub value: Value,
}

impl Token {
    pub(crate) fn new(kind: Kind) -> Self {
        Self {
            kind,
            value: Value::None,
        }
    }
}

pub(super) fn keyword(word: &[u8]) -> Option<Kind> {
    Some(match word {
        b"and" => Kind::And,
        b"break" => Kind::Break,
        b"do" => Kind::Do,
        b"else" => Kind::Else,
        b"elseif" => Kind::Elseif,
        b"end" => Kind::End,
        b"false" => Kind::False,
        b"for" => Kind::For,
        b"function" => Kind::Function,
        b"goto" => Kind::Goto,
        b"if" => Kind::If,
        b"in" => Kind::In,
        b"local" => Kind::Local,
        b"nil" => Kind::Nil,
        b"not" => Kind::Not,
        b"or" => Kind::Or,
        b"repeat" => Kind::Repeat,
        b"return" => Kind::Return,
        b"then" => Kind::Then,
        b"true" => Kind::True,
        b"until" => Kind::Until,
        b"while" => Kind::While,
        _ => return None,
    })
}

pub(crate) fn describe(kind: Kind) -> String {
    match kind {
        Kind::Char(byte) if (0x20..0x7f).contains(&byte) => format!("'{}'", char::from(byte)),
        Kind::Char(byte) => format!("'<\\{byte}>'"),
        Kind::Eos => "<eof>".to_string(),
        Kind::Float => "<number>".to_string(),
        Kind::Int => "<integer>".to_string(),
        Kind::Name => "<name>".to_string(),
        Kind::Str => "<string>".to_string(),
        other => format!("'{}'", fixed_text(other)),
    }
}

fn fixed_text(kind: Kind) -> &'static str {
    match kind {
        Kind::And => "and",
        Kind::Break => "break",
        Kind::Do => "do",
        Kind::Else => "else",
        Kind::Elseif => "elseif",
        Kind::End => "end",
        Kind::False => "false",
        Kind::For => "for",
        Kind::Function => "function",
        Kind::Goto => "goto",
        Kind::If => "if",
        Kind::In => "in",
        Kind::Local => "local",
        Kind::Nil => "nil",
        Kind::Not => "not",
        Kind::Or => "or",
        Kind::Repeat => "repeat",
        Kind::Return => "return",
        Kind::Then => "then",
        Kind::True => "true",
        Kind::Until => "until",
        Kind::While => "while",
        Kind::IntDiv => "//",
        Kind::Concat => "..",
        Kind::Dots => "...",
        Kind::EqEq => "==",
        Kind::GreaterEq => ">=",
        Kind::LessEq => "<=",
        Kind::NotEq => "~=",
        Kind::Shl => "<<",
        Kind::Shr => ">>",
        Kind::DbColon => "::",
        Kind::Eos | Kind::Float | Kind::Int | Kind::Name | Kind::Str | Kind::Char(_) => "",
    }
}

pub(super) fn quoted(text: &[u8]) -> Vec<u8> {
    let end = text.iter().position(|&byte| byte == 0).unwrap_or(text.len());
    [&b"'"[..], &text[..end], &b"'"[..]].concat()
}

pub(super) fn hex_value(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        b'a'..=b'f' => digit - b'a' + 10,
        b'A'..=b'F' => digit - b'A' + 10,
        _ => 0,
    }
}
