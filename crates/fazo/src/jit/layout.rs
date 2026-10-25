use std::mem::offset_of;

use crate::{
    arena::Ref,
    object::{Captured, Closure, Frame, LuaFrame, Thread, UpVal},
    table::{Table, offsets},
    value::Value,
};

/// Byte offsets of a table's fields. Layouts Rust does not declare are found by probing known values, and a build where
/// probing fails compiles no inline field access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TableLayout {
    pub(crate) size: i64,
    pub(crate) shaped: i32,
    pub(crate) shape: i32,
    pub(crate) slots: i32,
    pub(crate) slots_len: i32,
    pub(crate) slots_capacity: i32,
    pub(crate) array: i32,
    pub(crate) array_len: i32,
    pub(crate) nodes: i32,
    pub(crate) nodes_len: i32,
    pub(crate) string_keys: i32,
    pub(crate) absent: i32,
    pub(crate) metatable_tag: i32,
    pub(crate) metatable_none: u32,
    pub(crate) metatable_index: i32,
}

impl TableLayout {
    pub(crate) fn probe() -> Option<Self> {
        let (vec_data, vec_len) = vec_layout()?;
        let vec_capacity = vec_capacity_offset(vec_data, vec_len)?;
        let (tag, none, index) = option_ref_layout()?;
        Some(Self {
            size: i64::try_from(std::mem::size_of::<Table>()).ok()?,
            shaped: i32::try_from(offsets::SHAPED).ok()?,
            shape: i32::try_from(offsets::SHAPE).ok()?,
            slots: i32::try_from(offsets::SLOTS + vec_data).ok()?,
            slots_len: i32::try_from(offsets::SLOTS + vec_len).ok()?,
            slots_capacity: i32::try_from(offsets::SLOTS + vec_capacity).ok()?,
            array: i32::try_from(offsets::ARRAY + vec_data).ok()?,
            array_len: i32::try_from(offsets::ARRAY + vec_len).ok()?,
            nodes: i32::try_from(offsets::NODES + vec_data).ok()?,
            nodes_len: i32::try_from(offsets::NODES + vec_len).ok()?,
            string_keys: i32::try_from(std::mem::offset_of!(crate::Vm, heap) + std::mem::offset_of!(crate::heap::Heap, string_keys) + vec_data).ok()?,
            absent: i32::try_from(offsets::ABSENT).ok()?,
            metatable_tag: i32::try_from(offsets::METATABLE + tag).ok()?,
            metatable_none: none,
            metatable_index: i32::try_from(offsets::METATABLE + index).ok()?,
        })
    }
}

/// Byte offsets for direct calls. The `#[repr(C)]` frame types have declared offsets; the vectors are probed like the table's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameLayout {
    pub(crate) stack_len: i32,
    pub(crate) frames_data: i32,
    pub(crate) frames_len: i32,
    pub(crate) frames_capacity: i32,
    pub(crate) frame_size: i64,
    pub(crate) lua: i32,
    pub(crate) closure_upvalues: i32,
}

impl FrameLayout {
    pub(crate) fn probe() -> Option<Self> {
        let (data, len) = vec_layout()?;
        let (frame_data, frame_len) = frame_vec_layout()?;
        if (frame_data, frame_len) != (data, len) {
            return None;
        }
        let capacity = (0..VEC_WORDS).map(|word| word * WORD).find(|&offset| offset != data && offset != len)?;
        let (upvalue_data, _) = upvalue_vec_layout()?;
        Some(Self {
            stack_len: i32::try_from(offset_of!(Thread, stack) + len).ok()?,
            frames_data: i32::try_from(offset_of!(Thread, frames) + data).ok()?,
            frames_len: i32::try_from(offset_of!(Thread, frames) + len).ok()?,
            frames_capacity: i32::try_from(offset_of!(Thread, frames) + capacity).ok()?,
            frame_size: i64::try_from(std::mem::size_of::<Frame>()).ok()?,
            lua: i32::try_from(std::mem::align_of::<Frame>()).ok()?,
            closure_upvalues: i32::try_from(offset_of!(Closure, upvalues) + upvalue_data).ok()?,
        })
    }

    pub(crate) fn field(self, offset: usize) -> i32 {
        self.lua + offset as i32
    }
}

/// The pc and status word share one 64-bit word, so a push writes both with one store; so do the extra-argument count and the pending operand.
pub(crate) mod lua_frame {
    use super::{LuaFrame, offset_of};
    use crate::object::status;

    pub(crate) const CLOSURE: usize = offset_of!(LuaFrame, closure);
    pub(crate) const BASE: usize = offset_of!(LuaFrame, base);
    pub(crate) const PC: usize = LuaFrame::PC_OFFSET;
    pub(crate) const STATUS: usize = LuaFrame::STATUS_OFFSET;
    pub(crate) const STATUS_SHIFT: u8 = 32;
    const _: () = assert!(STATUS == PC + 4);
    pub(crate) const NVARARGS: usize = LuaFrame::NVARARGS_OFFSET;
    /// Cleared by a push or an in-place tail call, so the interpreter fills them when it runs the frame.
    pub(crate) const CACHE: usize = offset_of!(LuaFrame, code);
    pub(crate) const CACHE_WORDS: usize = 2;
    const _: () = assert!(offset_of!(LuaFrame, upvalues) == CACHE + std::mem::size_of::<usize>());
    pub(crate) const TAIL: u32 = status::TAIL;
    pub(crate) const TAIL_KEEPS: u32 = status::NRESULTS | status::COUNTED | status::NESTED;
    /// Frames with these bits cannot have their return taken in place by compiled code.
    pub(crate) const OWN_CALLER: u32 = status::COUNTED | status::HIDDEN;
}

const WORD: usize = std::mem::size_of::<usize>();
const VEC_WORDS: usize = 3;

/// The offsets of the buffer pointer and the length in a `Vec<Value>`: the one word of the vector
/// that equals its `as_ptr`, and the one that equals its length, the same in two vectors of
/// different lengths and capacities.
fn vec_layout() -> Option<(usize, usize)> {
    let mut first: Vec<Value> = Vec::with_capacity(5);
    first.push(Value::Nil);
    let mut second: Vec<Value> = Vec::with_capacity(9);
    second.extend([Value::Nil, Value::Nil]);
    vectors_layout(&first, &second)
}

fn vec_capacity_offset(data: usize, len: usize) -> Option<usize> {
    let mut probe: Vec<Value> = Vec::with_capacity(7);
    probe.push(Value::Nil);
    let capacity = matching_word(&probe, probe.capacity())? * WORD;
    (capacity != data && capacity != len).then_some(capacity)
}

fn frame_vec_layout() -> Option<(usize, usize)> {
    let frame = || Frame::Lua(LuaFrame::called(Ref::new(0, 0), 0, 0, 0));
    let mut first: Vec<Frame> = Vec::with_capacity(5);
    first.push(frame());
    let mut second: Vec<Frame> = Vec::with_capacity(9);
    second.extend([frame(), frame()]);
    vectors_layout(&first, &second)
}

fn upvalue_vec_layout() -> Option<(usize, usize)> {
    let mut first: Vec<Captured> = Vec::with_capacity(5);
    first.push(Captured::default());
    let mut second: Vec<Captured> = Vec::with_capacity(9);
    second.extend([Captured::default(), Captured::default()]);
    vectors_layout(&first, &second)
}

/// Offsets for upvalue access: the closure's upvalue list cell, the `UpVal` variable pointer and the tag byte of its state.
pub(crate) mod upvalue {
    use super::{Captured, UpVal, offset_of};

    pub(crate) const VALUE: i32 = offset_of!(UpVal, value) as i32;
    pub(crate) const TAG: i32 = offset_of!(UpVal, state) as i32;
    pub(crate) const OPEN: i64 = 0;
    const _: () = assert!(VALUE == 0 && TAG == 8);

    pub(crate) fn cell(index: u8) -> i64 {
        (i64::from(index) * std::mem::size_of::<Captured>() as i64) + offset_of!(Captured, cell) as i64
    }
}

fn vectors_layout<T>(first: &Vec<T>, second: &Vec<T>) -> Option<(usize, usize)> {
    if std::mem::size_of::<Vec<T>>() != VEC_WORDS * WORD {
        return None;
    }
    let data = matching_word(first, first.as_ptr() as usize)?;
    let len = matching_word(first, first.len())?;
    let same = matching_word(second, second.as_ptr() as usize)? == data && matching_word(second, second.len())? == len;
    (same && data != len).then_some((data * WORD, len * WORD))
}

fn matching_word<T>(vector: &Vec<T>, expected: usize) -> Option<usize> {
    debug_assert_eq!(std::mem::size_of::<Vec<T>>(), VEC_WORDS * WORD);
    // SAFETY: `vectors_layout` checked that a `Vec<T>` is three words, all initialized, so copying its bytes as `usize` is sound.
    let words: [usize; VEC_WORDS] = unsafe { std::mem::transmute_copy(vector) };
    let mut matching = (0..VEC_WORDS).filter(|&word| words[word] == expected);
    let found = matching.next()?;
    matching.next().is_none().then_some(found)
}

const OPTION_WORDS: usize = 3;
const WORD32: usize = std::mem::size_of::<u32>();

/// `Ref` has no niche, so `Option<Ref<Table>>` carries a 32-bit discriminant word; its offsets are found by probing.
fn option_ref_layout() -> Option<(usize, u32, usize)> {
    if std::mem::size_of::<Option<Ref<Table>>>() != OPTION_WORDS * WORD32 {
        return None;
    }
    let first = option_words(Some(Ref::new(0x1357_9bdf, 0x2468_ace0)));
    let second = option_words(Some(Ref::new(0x0fed_cba9, 0x7654_3210)));
    let index = (0..OPTION_WORDS - 1).find(|&word| first[word] == 0x1357_9bdf && first[word + 1] == 0x2468_ace0)?;
    if second[index] != 0x0fed_cba9 || second[index + 1] != 0x7654_3210 {
        return None;
    }
    let tag = (0..OPTION_WORDS).find(|&word| word != index && word != index + 1)?;
    let none = option_words(None)[tag];
    (none != first[tag] && first[tag] == second[tag]).then_some((tag * WORD32, none, index * WORD32))
}

fn option_words(option: Option<Ref<Table>>) -> [u32; OPTION_WORDS] {
    let mut words = [0u32; OPTION_WORDS];
    let storage = std::ptr::from_mut(&mut words).cast::<Option<Ref<Table>>>();
    debug_assert!(storage.is_aligned());
    // SAFETY: `option_ref_layout` checked that the option is three 32-bit words, aligned for them; zeroed so `None` reads as zero.
    unsafe {
        storage.write(option);
        words
    }
}
