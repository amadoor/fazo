//! Eager deopt exits, after Maglev and V8's `TranslationArray` (src/deoptimizer/translation-array.h): the runtime rebuilds frame state from a side table of value locations.

use super::super::{
    DeoptPoint,
    ir::{FrameState, Val},
};
use crate::{jit::JitCtx, value::Value};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Word {
    /// A register, by its index in the dump.
    Reg(u8),
    /// A spill slot, by its offset from the frame's stack pointer.
    Slot(i32),
    Imm(u64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Place {
    Boxed { tag: Word, payload: Word },
    Tagged { tag: u8, payload: Word },
}

pub(crate) struct Exit {
    pub(crate) state: u32,
    start: u32,
    len: u32,
}

/// Owned by the unit and addressed by its code, so it never moves once the code exists.
#[derive(Default)]
pub(crate) struct ExitTable {
    pub(crate) exits: Vec<Exit>,
    pub(crate) places: Vec<(Val, Place)>,
    pub(crate) states: Vec<FrameState>,
    pub(crate) points: Box<[DeoptPoint]>,
}

impl ExitTable {
    pub(crate) fn add(&mut self, state: u32, mut places: Vec<(Val, Place)>) -> u32 {
        places.sort_unstable_by_key(|&(value, _)| value);
        places.dedup_by_key(|&mut (value, _)| value);
        if let Some(last) = self.exits.last()
            && last.state == state
            && self.places[last.start as usize..(last.start + last.len) as usize] == places[..]
        {
            return self.exits.len() as u32 - 1;
        }
        let start = self.places.len() as u32;
        let len = places.len() as u32;
        self.places.extend(places);
        self.exits.push(Exit { state, start, len });
        self.exits.len() as u32 - 1
    }

    fn place(&self, exit: &Exit, value: Val) -> Option<Place> {
        let places = &self.places[exit.start as usize..(exit.start + exit.len) as usize];
        let index = places.binary_search_by_key(&value, |&(known, _)| known).ok()?;
        Some(places[index].1)
    }
}

pub(crate) type ExitFn = extern "C" fn(*mut JitCtx, *const ExitTable, u32, *const u64, *const u8) -> i64;

/// Rebuilds the frame and deoptimizes it; answers the code the compiled function returns.
pub(crate) extern "C" fn exit(ctx: *mut JitCtx, table: *const ExitTable, index: u32, dump: *const u64, frame: *const u8) -> i64 {
    // SAFETY: the code passes its unit's table, which outlives the code, and an exit index the table holds.
    let table = unsafe { &*table };
    let exit = &table.exits[index as usize];
    with_vm(ctx, |ctx, vm| {
        store_state(ctx, vm, table, exit, (dump, frame));
        vm.jit_deopt(ctx, &table.points[exit.state as usize]).wrapping_neg()
    })
}

/// Writes the frame state's registers back, for a safe point or a collector step that must see them.
pub(crate) extern "C" fn store(ctx: *mut JitCtx, table: *const ExitTable, index: u32, dump: *const u64, frame: *const u8) -> i64 {
    // SAFETY: as in `exit`.
    let table = unsafe { &*table };
    let exit = &table.exits[index as usize];
    with_vm(ctx, |ctx, vm| store_state(ctx, vm, table, exit, (dump, frame)));
    0
}

fn with_vm<T>(ctx: *mut JitCtx, body: impl FnOnce(&mut JitCtx, &mut crate::Vm) -> T) -> T {
    // SAFETY: compiled code passes the context the driver built for this frame, which outlives the call, and its VM
    // pointer names the VM running the code on this thread.
    let ctx = unsafe { &mut *ctx };
    // SAFETY: as above; nothing else holds the VM while compiled code runs.
    let vm = unsafe { &mut *ctx.vm };
    if ctx.pending != 0 {
        vm.jit_push_pending(ctx);
    }
    body(ctx, vm)
}

fn read(word: Word, (dump, frame): (*const u64, *const u8)) -> u64 {
    match word {
        Word::Imm(bits) => bits,
        Word::Reg(index) => {
            debug_assert!(!dump.is_null());
            // SAFETY: the exit tail stored every allocatable register in the dump, which lives until the runtime returns.
            unsafe { dump.add(usize::from(index)).read() }
        }
        Word::Slot(offset) => {
            debug_assert!(!frame.is_null());
            // SAFETY: the offset names a spill slot of the exiting frame, which is below it on the stack and unchanged.
            unsafe { frame.offset(offset as isize).cast::<u64>().read_unaligned() }
        }
    }
}

fn words(place: Place, sources: (*const u64, *const u8)) -> [u64; 2] {
    match place {
        Place::Boxed { tag, payload } => [read(tag, sources) & 0xff, read(payload, sources)],
        Place::Tagged { tag, payload } => [u64::from(tag), read(payload, sources)],
    }
}

fn value(words: [u64; 2]) -> Value {
    debug_assert!(words[0] <= 10);
    // SAFETY: `Value` is `#[repr(C, u8)]`, 16 bytes with the tag at 0 and the payload at 8, and the words are a tag and
    // payload compiled code took from a valid value of that kind.
    unsafe { std::mem::transmute::<[u64; 2], Value>(words) }
}

/// Sunk tables are built last, after every other register is in place, as `lower.rs` stores a frame state.
fn store_state(ctx: &mut JitCtx, vm: &mut crate::Vm, table: &ExitTable, exit: &Exit, sources: (*const u64, *const u8)) {
    let state = &table.states[exit.state as usize];
    let lookup = |value: Val| words(table.place(exit, value).expect("an exit places every value of its frame state"), sources);
    for &(slot, held) in &state.slots {
        if state.object(held).is_some() {
            continue;
        }
        write(ctx, slot, lookup(held));
    }
    for &(slot, held) in &state.slots {
        let Some(object) = state.object(held) else {
            continue;
        };
        let meta = object.meta.map_or(Value::Nil, |meta| value(lookup(meta)));
        let fields: Vec<Value> = object.fields.iter().map(|&field| value(lookup(field))).collect();
        let shape = crate::arena::Ref::new(object.shape as u32, (object.shape >> 32) as u32);
        let built = vm.jit_materialize(shape, object.hash_log, meta, &fields);
        vm.jit_refresh_context(ctx);
        let payload = match object.original {
            Some((flag, original)) if lookup(flag)[1] & 0xff == 0 => lookup(original)[1],
            _ => built.packed(),
        };
        write(ctx, slot, [5, payload]);
    }
}

fn write(ctx: &JitCtx, slot: u32, [tag, payload]: [u64; 2]) {
    debug_assert!(!ctx.registers.is_null());
    // SAFETY: the frame state names registers of the compiled frame, which the driver sized its stack window for.
    unsafe {
        let at = ctx.registers.add(slot as usize).cast::<u64>();
        at.write(tag);
        at.add(1).write(payload);
    }
}
