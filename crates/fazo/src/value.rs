use crate::{
    arena::Ref,
    object::{Closure, LuaString, Native, Thread, Userdata},
    table::Table,
};

/// Holds up to four values inline without allocating.
pub type Values = smallvec::SmallVec<[Value; 4]>;

#[macro_export]
macro_rules! values {
    ($($value:expr),* $(,)?) => {
        $crate::Values::from_slice(&[$($value),*])
    };
}

/// A Lua value; heap objects are referenced by index, so the value is `Copy` and holds no pointers.
/// Stored as 16 bytes with the tag at offset 0.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[repr(C, u8)]
pub enum Value {
    Nil = 0,
    Boolean(bool) = 1,
    Integer(i64) = 2,
    Float(f64) = 3,
    String(Ref<LuaString>) = 4,
    Table(Ref<Table>) = 5,
    Closure(Ref<Closure>) = 6,
    Native(Ref<Native>) = 7,
    Thread(Ref<Thread>) = 8,
    Userdata(Ref<Userdata>) = 9,
    LightUserdata(usize) = 10,
}

const _: () = assert!(std::mem::size_of::<Value>() == 16);

impl Value {
    #[inline(always)]
    pub(crate) fn is_collectable(self) -> bool {
        matches!(self, Value::String(_) | Value::Table(_) | Value::Closure(_) | Value::Native(_) | Value::Thread(_) | Value::Userdata(_))
    }

    pub fn is_nil(self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn is_falsy(self) -> bool {
        matches!(self, Value::Nil | Value::Boolean(false))
    }

    pub fn type_name(self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Boolean(_) => "boolean",
            Value::Integer(_) | Value::Float(_) => "number",
            Value::String(_) => "string",
            Value::Table(_) => "table",
            Value::Closure(_) | Value::Native(_) => "function",
            Value::Thread(_) => "thread",
            Value::Userdata(_) | Value::LightUserdata(_) => "userdata",
        }
    }

    pub fn is_number(self) -> bool {
        matches!(self, Value::Integer(_) | Value::Float(_))
    }

    pub fn is_function(self) -> bool {
        matches!(self, Value::Closure(_) | Value::Native(_))
    }
}
