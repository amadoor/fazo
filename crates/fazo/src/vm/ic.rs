//! Follows V8's inline caches and feedback vectors, and LuaJIT's table access templates (both permissively licensed).

use super::{Event, quick, window::Window};
use crate::{
    Vm,
    arena::Ref,
    gc::Gc,
    object::LuaString,
    proto::Proto,
    shape::Shape,
    table::Table,
    value::Value,
};

pub(crate) const ABSENT: u32 = u32::MAX;

const WAYS: usize = 4;
const MEGAMORPHIC_ENTRIES: usize = 256;

/// `next` is the shape after a store: unchanged for an existing slot, the child shape when the key is added.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) shape: Ref<Shape>,
    pub(crate) next: Ref<Shape>,
    pub(crate) slot: u32,
}

impl Entry {
    pub(crate) fn plain(shape: Ref<Shape>, slot: u32) -> Self {
        Self { shape, next: shape, slot }
    }

    pub(crate) fn adds_key(&self) -> bool {
        self.next != self.shape
    }
}

/// Checked at every hit, so a stale entry falls back to the generic lookup. The metatable is not part of the entry, so objects with the same shape and class share one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MethodEntry {
    pub(crate) receiver: Ref<Shape>,
    pub(crate) metatable_shape: Ref<Shape>,
    pub(crate) index_slot: u32,
    pub(crate) class: Ref<Table>,
    pub(crate) class_shape: Ref<Shape>,
    pub(crate) class_slot: u32,
    pub(crate) depth: u8,
    pub(crate) deeper: [Hop; DEEPER_HOPS],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Hop {
    pub(crate) metatable_shape: Ref<Shape>,
    pub(crate) index_slot: u32,
    pub(crate) class: Ref<Table>,
    pub(crate) class_shape: Ref<Shape>,
}

pub(crate) const DEEPER_HOPS: usize = 2;

impl MethodEntry {
    pub(crate) fn hops(&self) -> &[Hop] {
        &self.deeper[..usize::from(self.depth)]
    }

    fn shapes(&self) -> impl Iterator<Item = Ref<Shape>> + '_ {
        [self.receiver, self.metatable_shape, self.class_shape]
            .into_iter()
            .chain(self.hops().iter().flat_map(|hop| [hop.metatable_shape, hop.class_shape]))
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) enum InlineCache {
    #[default]
    Uninitialized,
    Monomorphic(Entry),
    Polymorphic(Box<[Entry]>),
    Method(Box<[MethodEntry]>),
    Callee(Callee),
    Callees(Box<[Callee]>),
    Megamorphic,
}

/// Keyed by prototype, so a site calling fresh closures of one prototype stays monomorphic.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Callee {
    pub(crate) value: Value,
    pub(crate) proto: Option<Ref<Proto>>,
}

impl Callee {
    fn same(self, other: Self) -> bool {
        same_callee(self.value, other.value) || (self.proto.is_some() && self.proto == other.proto)
    }
}

impl InlineCache {
    pub(crate) fn find(&self, shape: Ref<Shape>) -> Option<Entry> {
        match self {
            Self::Monomorphic(entry) => (entry.shape == shape).then_some(*entry),
            Self::Polymorphic(entries) => entries.iter().find(|entry| entry.shape == shape).copied(),
            Self::Uninitialized | Self::Method(_) | Self::Callee(_) | Self::Callees(_) | Self::Megamorphic => None,
        }
    }

    pub(crate) fn record_method(&mut self, method: MethodEntry) {
        *self = match std::mem::take(self) {
            Self::Method(methods) if methods.len() < WAYS => {
                if methods.iter().any(|known| known.receiver == method.receiver && known.metatable_shape == method.metatable_shape && known.class == method.class && known.depth == method.depth) {
                    Self::Method(methods)
                } else {
                    let mut grown = methods.into_vec();
                    grown.push(method);
                    Self::Method(grown.into_boxed_slice())
                }
            }
            Self::Method(methods) => Self::Method(methods),
            _ => Self::Method(Box::new([method])),
        };
    }

    /// Shapes are kept alive with the prototype, so a site's transitions outlive the objects that took them.
    pub(crate) fn push_shapes(&self, shapes: &mut Vec<Gc>) {
        match self {
            Self::Monomorphic(entry) => shapes.extend([Gc::Shape(entry.shape), Gc::Shape(entry.next)]),
            Self::Polymorphic(entries) => shapes.extend(entries.iter().flat_map(|entry| [Gc::Shape(entry.shape), Gc::Shape(entry.next)])),
            Self::Method(methods) => shapes.extend(methods.iter().flat_map(MethodEntry::shapes).map(Gc::Shape)),
            Self::Uninitialized | Self::Callee(_) | Self::Callees(_) | Self::Megamorphic => {}
        }
    }

    fn state(&self) -> (u8, usize) {
        match self {
            Self::Uninitialized => (0, 0),
            Self::Monomorphic(_) => (1, 1),
            Self::Polymorphic(entries) => (2, entries.len()),
            Self::Method(methods) => (3, methods.len()),
            Self::Callee(_) => (4, 1),
            Self::Callees(callees) => (5, callees.len()),
            Self::Megamorphic => (6, 0),
        }
    }

    pub(crate) fn is_megamorphic(&self) -> bool {
        matches!(self, Self::Megamorphic)
    }

    /// Only Lua closures and natives have identity to record.
    #[inline(never)]
    pub(crate) fn record_callee(&mut self, callee: Callee) {
        if !matches!(callee.value, Value::Closure(_) | Value::Native(_)) {
            return;
        }
        *self = match std::mem::take(self) {
            Self::Uninitialized => Self::Callee(callee),
            Self::Callee(previous) if previous.same(callee) => Self::Callee(callee),
            Self::Callee(previous) => Self::Callees(Box::new([previous, callee])),
            Self::Callees(mut callees) if let Some(index) = callees.iter().position(|known| known.same(callee)) => {
                callees[index] = callee;
                Self::Callees(callees)
            }
            Self::Callees(callees) if callees.len() < WAYS => {
                let mut grown = callees.into_vec();
                grown.push(callee);
                Self::Callees(grown.into_boxed_slice())
            }
            Self::Callees(_) | Self::Megamorphic => Self::Megamorphic,
            other @ (Self::Monomorphic(_) | Self::Polymorphic(_) | Self::Method(_)) => other,
        };
    }

    fn record(&mut self, entry: Entry) {
        *self = match std::mem::take(self) {
            Self::Uninitialized => Self::Monomorphic(entry),
            Self::Monomorphic(previous) => Self::Polymorphic(Box::new([previous, entry])),
            Self::Polymorphic(entries) if entries.len() < WAYS => {
                let mut grown = entries.into_vec();
                grown.push(entry);
                Self::Polymorphic(grown.into_boxed_slice())
            }
            Self::Polymorphic(_) | Self::Megamorphic => Self::Megamorphic,
            other @ (Self::Callee(_) | Self::Callees(_) | Self::Method(_)) => other,
        };
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn same_callee(left: Value, right: Value) -> bool {
    match (left, right) {
        (Value::Closure(a), Value::Closure(b)) => a == b,
        (Value::Native(a), Value::Native(b)) => a == b,
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct MegamorphicEntry {
    shape: Ref<Shape>,
    key: Ref<LuaString>,
    slot: u32,
}

/// Direct-mapped, so a fixed size bounds it; the arena's generation keeps a reused shape from matching a stale entry.
pub(crate) struct MegamorphicCache {
    entries: Vec<Option<MegamorphicEntry>>,
}

impl Default for MegamorphicCache {
    fn default() -> Self {
        Self {
            entries: vec![None; MEGAMORPHIC_ENTRIES],
        }
    }
}

impl MegamorphicCache {
    fn position(shape: Ref<Shape>, key: Ref<LuaString>) -> usize {
        let mixed = (shape.index() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ key.index() as u64;
        (mixed >> 32) as usize % MEGAMORPHIC_ENTRIES
    }

    fn find(&self, shape: Ref<Shape>, key: Ref<LuaString>) -> Option<u32> {
        self.entries[Self::position(shape, key)]
            .filter(|entry| entry.shape == shape && entry.key == key)
            .map(|entry| entry.slot)
    }

    fn store(&mut self, shape: Ref<Shape>, key: Ref<LuaString>, slot: u32) {
        self.entries[Self::position(shape, key)] = Some(MegamorphicEntry { shape, key, slot });
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Probe {
    Value(Value),
    Absent(Ref<Table>),
    Unknown,
}

pub(crate) enum Cached {
    Value(Value),
    Absent(Ref<Table>),
    Unknown,
}

/// Calls before a loop-free prototype gets feedback; a prototype with a loop gets it at build time, since one call runs its sites many times.
pub(crate) const FEEDBACK_CALLS: u32 = 2;

pub(crate) fn initial_feedback(code: &[crate::instruction::Instruction]) -> Vec<InlineCache> {
    let has_loop = code.iter().enumerate().any(|(index, instruction)| match *instruction {
        crate::instruction::Instruction::ForLoop { .. } | crate::instruction::Instruction::TForLoop { .. } => true,
        crate::instruction::Instruction::Jmp { sj } => sj < 0 && index > 0,
        _ => false,
    });
    if has_loop {
        vec![InlineCache::Uninitialized; code.len()]
    } else {
        Vec::new()
    }
}

impl Vm {

    #[inline]
    pub(crate) fn cached_read(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Ref<LuaString>) -> Cached {
        let Value::Table(table) = object else {
            return Cached::Unknown;
        };
        let Some(shape) = self.heap.tables.live(table).shape() else {
            return Cached::Unknown;
        };
        let entry = match self.heap.protos.live(proto).exec.feedback[site].find(shape) {
            Some(entry) => entry,
            None => self.read_miss(proto, site, shape, key),
        };
        if entry.slot == ABSENT {
            return Cached::Absent(table);
        }
        match self.heap.tables.live(table).slot_value(entry.slot as usize) {
            Value::Nil => Cached::Unknown,
            value => Cached::Value(value),
        }
    }

    /// A megamorphic site asks the VM's shared cache first.
    fn read_miss(&mut self, proto: Ref<Proto>, site: usize, shape: Ref<Shape>, key: Ref<LuaString>) -> Entry {
        let megamorphic = self.heap.protos.live(proto).exec.feedback[site].is_megamorphic();
        let slot = match megamorphic.then(|| self.megamorphic.find(shape, key)).flatten() {
            Some(slot) => slot,
            None => {
                let slot = self.slot_in_shape(shape, key).map_or(ABSENT, |slot| slot as u32);
                if megamorphic {
                    self.megamorphic.store(shape, key, slot);
                }
                slot
            }
        };
        let entry = Entry::plain(shape, slot);
        self.record_entry(proto, site, entry);
        entry
    }

    /// Notifies the tiers when the site's state changes.
    fn record_entry(&mut self, proto: Ref<Proto>, site: usize, entry: Entry) {
        let cache = &mut self.heap.protos.live_mut(proto).exec.feedback[site];
        let before = cache.state();
        cache.record(entry);
        let changed = cache.state() != before;
        self.keep_site_shapes(proto, &[entry.shape, entry.next]);
        if changed {
            self.quicken_site(proto, site);
            self.jit_feedback_changed(proto);
        }
    }

    fn slot_in_shape(&self, shape: Ref<Shape>, key: Ref<LuaString>) -> Option<usize> {
        let hash = self.heap.key_view().hash(key);
        self.heap.shapes.live(shape).slot(&self.heap.shapes, key, hash)
    }

    /// False when a nil slot may trigger `__newindex`, so the store takes the generic path.
    pub(crate) fn cached_store(&mut self, proto: Ref<Proto>, site: usize, object: Value, value: Value) -> bool {
        let Value::Table(table) = object else {
            return false;
        };
        let Some(shape) = self.heap.tables.live(table).shape() else {
            return false;
        };
        let Some(entry) = self.heap.protos.live(proto).exec.feedback[site].find(shape) else {
            return false;
        };
        let target = self.heap.tables.live(table);
        if target.metatable.is_some() {
            let present = !entry.adds_key() && entry.slot != ABSENT && !target.slot_value(entry.slot as usize).is_nil();
            if !present && self.blocks_new_index(table) {
                return false;
            }
        }
        if entry.adds_key() && !self.heap.take_transition(table, entry.next) {
            return false;
        }
        self.heap.tables.live_mut(table).set_slot(entry.slot as usize, value);
        self.heap.table_barrier_back(table, value);
        true
    }

    fn blocks_new_index(&mut self, table: Ref<Table>) -> bool {
        match self.heap.tables.live(table).metatable {
            Some(metatable) => !self.table_event(metatable, crate::vm::Event::NewIndex).is_nil(),
            None => false,
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn site_feedback(&self, proto: Ref<Proto>, site: usize) -> Option<&InlineCache> {
        self.heap.protos.live(proto).exec.feedback.get(site)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn site_feedback_mut(&mut self, proto: Ref<Proto>, site: usize) -> Option<&mut InlineCache> {
        self.heap.protos.live_mut(proto).exec.feedback.get_mut(site)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn record_call(&mut self, proto: Ref<Proto>, site: usize, callee: Value) {
        if self.options.tier == crate::Tier::Interpreter {
            return;
        }
        let Some(cache) = self.site_feedback_mut(proto, site) else {
            return;
        };
        if matches!(cache, InlineCache::Callee(known) if same_callee(known.value, callee)) {
            return;
        }
        self.record_new_callee(proto, site, callee);
    }

    /// A closure of an already recorded prototype takes its place, since compiled code enters every closure of that prototype.
    #[inline(never)]
    fn record_new_callee(&mut self, proto: Ref<Proto>, site: usize, callee: Value) {
        let recorded = match self.site_feedback(proto, site) {
            Some(InlineCache::Callees(known)) => known.iter().any(|known| same_callee(known.value, callee)),
            Some(InlineCache::Megamorphic) => true,
            _ => false,
        };
        if recorded {
            return;
        }
        let called = match callee {
            Value::Closure(closure) => Some(self.heap.closures.live(closure).proto),
            _ => None,
        };
        let Some(cache) = self.site_feedback_mut(proto, site) else {
            return;
        };
        let before = cache.state();
        cache.record_callee(Callee { value: callee, proto: called });
        if cache.state() != before {
            self.jit_feedback_changed(proto);
        }
    }

    /// Metamethod keys are not recorded, since a cached store would skip clearing the table's absence flags.
    pub(crate) fn record_store(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Ref<LuaString>, before: Option<Ref<Shape>>) {
        let (Value::Table(table), Some(before)) = (object, before) else {
            return;
        };
        if self.heap.bytes(key).starts_with(b"__") {
            return;
        }
        let Some(after) = self.heap.tables.live(table).shape() else {
            return;
        };
        let hash = self.heap.key_view().hash(key);
        let Some(slot) = self.heap.shapes.live(after).slot(&self.heap.shapes, key, hash) else {
            return;
        };
        let entry = if after == before {
            Entry::plain(before, slot as u32)
        } else if self.heap.shapes.live(after).len() == slot + 1 && slot == self.heap.shapes.live(before).len() {
            Entry {
                shape: before,
                next: after,
                slot: slot as u32,
            }
        } else {
            return;
        };
        self.record_entry(proto, site, entry);
    }


    /// Write barrier for the shapes the feedback names; the collector marks them through the prototype.
    fn keep_site_shapes(&mut self, proto: Ref<Proto>, shapes: &[Ref<Shape>]) {
        for &shape in shapes {
            self.heap.barrier(Gc::Proto(proto), Some(Gc::Shape(shape)));
        }
    }

    pub(crate) fn shape_of_value(&self, object: Value) -> Option<Ref<Shape>> {
        match object {
            Value::Table(table) => self.heap.tables.live(table).shape(),
            _ => None,
        }
    }

    /// Chains longer than `1 + DEEPER_HOPS` `__index` tables stay on the generic path.
    pub(super) fn record_absent_chain(&mut self, proto: Ref<Proto>, site: usize, receiver: Ref<Table>, key: Ref<LuaString>) {
        if matches!(self.site_feedback(proto, site), Some(InlineCache::Method(methods)) if methods.len() >= WAYS) {
            return;
        }
        let Some(method) = self.absent_chain_entry(receiver, key) else {
            return;
        };
        let Some(cache) = self.site_feedback_mut(proto, site) else {
            return;
        };
        let before = cache.state();
        cache.record_method(method);
        let changed = cache.state() != before;
        let shapes: smallvec::SmallVec<[Ref<Shape>; 8]> = method.shapes().collect();
        self.keep_site_shapes(proto, &shapes);
        if changed {
            self.quicken_site(proto, site);
            self.jit_feedback_changed(proto);
        }
    }

    fn absent_chain_entry(&self, receiver: Ref<Table>, key: Ref<LuaString>) -> Option<MethodEntry> {
        let receiver_shape = self.heap.tables.live(receiver).shape()?;
        let first = self.index_hop(receiver)?;
        let mut method = MethodEntry {
            receiver: receiver_shape,
            metatable_shape: first.metatable_shape,
            index_slot: first.index_slot,
            class: first.class,
            class_shape: first.class_shape,
            class_slot: ABSENT,
            depth: 0,
            deeper: [first; DEEPER_HOPS],
        };
        let mut last = first;
        loop {
            if let Some(slot) = self.slot_in_shape(last.class_shape, key) {
                if self.heap.tables.live(last.class).slot_value(slot).is_nil() {
                    return None;
                }
                method.class_slot = slot as u32;
                return Some(method);
            }
            if usize::from(method.depth) == DEEPER_HOPS {
                return None;
            }
            last = self.index_hop(last.class)?;
            method.deeper[usize::from(method.depth)] = last;
            method.depth += 1;
        }
    }

    fn index_hop(&self, table: Ref<Table>) -> Option<Hop> {
        let metatable = self.heap.tables.live(table).metatable?;
        let metatable_shape = self.heap.tables.live(metatable).shape()?;
        let index_name = self.heap.events[Event::Index as usize];
        let index_slot = self.slot_in_shape(metatable_shape, index_name)?;
        let Value::Table(class) = self.heap.tables.live(metatable).slot_value(index_slot) else {
            return None;
        };
        let class_shape = self.heap.tables.live(class).shape()?;
        Some(Hop {
            metatable_shape,
            index_slot: index_slot as u32,
            class,
            class_shape,
        })
    }

    /// Transitions are not answered here.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn cached_slot_in(&self, cache: Option<&InlineCache>, shape: Ref<Shape>, key: Ref<LuaString>) -> Option<u32> {
        let cache = cache?;
        match cache.find(shape) {
            Some(entry) if !entry.adds_key() => Some(entry.slot),
            Some(_) => None,
            None if cache.is_megamorphic() => self.megamorphic.find(shape, key),
            None => None,
        }
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn field_probe(&self, proto: Ref<Proto>, site: usize, object: Value, key: Value) -> Probe {
        self.field_probe_in(self.site_feedback(proto, site), object, key)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn field_probe_in(&self, cache: Option<&InlineCache>, object: Value, key: Value) -> Probe {
        let (Value::Table(table), Value::String(string)) = (object, key) else {
            return Probe::Unknown;
        };
        let object = self.heap.tables.live(table);
        let Some(shape) = object.shape() else {
            return Probe::Unknown;
        };
        let Some(slot) = self.cached_slot_in(cache, shape, string) else {
            return Probe::Unknown;
        };
        if slot == ABSENT {
            return Probe::Absent(table);
        }
        match object.slot_value(slot as usize) {
            Value::Nil => Probe::Unknown,
            value => Probe::Value(value),
        }
    }

    /// Follows `luaH_getshortstr` in ltable.c; a long string key matches only the same string object.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn field_value_in<'c>(&self, cache: impl FnOnce() -> Option<&'c InlineCache>, object: Value, key: Value) -> &Value {
        static NIL: Value = Value::Nil;
        let (Value::Table(table), Value::String(string)) = (object, key) else {
            return &NIL;
        };
        let object = Window::table_object(self, table);
        let Some(shape) = object.shape() else {
            return object.node_short_str(string, Window::string_hash(self, string));
        };
        match self.cached_slot_in(cache(), shape, string) {
            Some(slot) if slot != ABSENT => &object.slot_values()[slot as usize],
            _ => &NIL,
        }
    }

    /// Nil leaves the read to the generic path.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn shape_hit(&self, cache: Option<&InlineCache>, object: Value) -> &Value {
        static NIL: Value = Value::Nil;
        let (Value::Table(table), Some(InlineCache::Monomorphic(entry))) = (object, cache) else {
            return &NIL;
        };
        let target = Window::table_object(self, table);
        if target.shape() != Some(entry.shape) {
            return &NIL;
        }
        Window::cached_slot(target, entry.slot as usize)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn absent_hit(&self, cache: Option<&InlineCache>, object: Value) -> bool {
        let (Value::Table(table), Some(InlineCache::Monomorphic(entry))) = (object, cache) else {
            return false;
        };
        debug_assert_eq!(entry.slot, ABSENT);
        self.heap.tables.live(table).shape() == Some(entry.shape) && self.absent_is_nil(table)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn shapes_hit(&self, cache: Option<&InlineCache>, object: Value) -> Value {
        let (Value::Table(table), Some(InlineCache::Polymorphic(entries))) = (object, cache) else {
            return Value::Nil;
        };
        let target = self.heap.tables.live(table);
        let Some(shape) = target.shape() else {
            return Value::Nil;
        };
        for entry in entries.iter() {
            if entry.shape == shape && entry.slot != ABSENT {
                return target.slot_value(entry.slot as usize);
            }
        }
        Value::Nil
    }

    /// No `__newindex` applies, since the slot holds a value.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn shape_store_hit(&mut self, cache: Option<&InlineCache>, object: Value, value: Value) -> bool {
        let (Value::Table(table), Some(InlineCache::Monomorphic(entry))) = (object, cache) else {
            return false;
        };
        let slot = entry.slot as usize;
        let target = self.heap.tables.live_mut(table);
        if target.shape() != Some(entry.shape) || target.slot_value(slot).is_nil() {
            return false;
        }
        target.set_slot(slot, value);
        if value.is_collectable() {
            self.heap.table_barrier_back(table, value);
        }
        true
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn transition_hit(&mut self, cache: Option<&InlineCache>, object: Value, value: Value) -> bool {
        let (Value::Table(table), Some(&InlineCache::Monomorphic(entry))) = (object, cache) else {
            return false;
        };
        if self.heap.tables.live(table).shape() != Some(entry.shape) {
            return false;
        }
        self.transition_store_hit(table, entry, value)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn method_hit(&self, proto: Ref<Proto>, site: usize, table: Ref<Table>) -> Option<Value> {
        self.method_hit_in(self.site_feedback(proto, site), table)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn method_hit_in(&self, cache: Option<&InlineCache>, table: Ref<Table>) -> Option<Value> {
        let Some(InlineCache::Method(methods)) = cache else {
            return None;
        };
        let receiver = self.heap.tables.live(table);
        let shape = receiver.shape()?;
        for method in methods.iter() {
            if method.receiver == shape
                && let Some(value) = self.method_value_of(method, receiver)
            {
                return Some(value);
            }
        }
        None
    }

    /// Any store of a metamethod key clears the absence flag, so the flag stays valid.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn absent_is_nil(&self, table: Ref<Table>) -> bool {
        match self.heap.tables.live(table).metatable {
            None => true,
            Some(metatable) => Event::Index.absence_bit().is_some_and(|bit| self.heap.tables.live(metatable).is_absent(bit)),
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn method_value_of(&self, method: &MethodEntry, receiver: &Table) -> Option<Value> {
        let metatable = self.heap.tables.live(receiver.metatable?);
        if metatable.shape() != Some(method.metatable_shape) {
            return None;
        }
        if !matches!(metatable.slot_value(method.index_slot as usize), Value::Table(class) if class == method.class) {
            return None;
        }
        let mut class = self.heap.tables.live(method.class);
        if class.shape() != Some(method.class_shape) {
            return None;
        }
        for hop in method.hops() {
            let metatable = self.heap.tables.live(class.metatable?);
            if metatable.shape() != Some(hop.metatable_shape) {
                return None;
            }
            if !matches!(metatable.slot_value(hop.index_slot as usize), Value::Table(next) if next == hop.class) {
                return None;
            }
            class = self.heap.tables.live(hop.class);
            if class.shape() != Some(hop.class_shape) {
                return None;
            }
        }
        match class.slot_value(method.class_slot as usize) {
            Value::Nil => None,
            value => Some(value),
        }
    }

    /// Only the write barrier for the value is needed, since the key is a shape key kept alive with the table.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn field_store_hit_in(&mut self, cache: Option<&InlineCache>, object: Value, key: Value, value: Value) -> bool {
        let (Value::Table(table), Value::String(string), Some(cache)) = (object, key, cache) else {
            return false;
        };
        let Some(shape) = self.heap.tables.live(table).shape() else {
            return false;
        };
        let entry = match cache.find(shape) {
            Some(entry) => entry,
            None if cache.is_megamorphic() => match self.megamorphic.find(shape, string) {
                Some(slot) => Entry::plain(shape, slot),
                None => return false,
            },
            None => return false,
        };
        if entry.slot == ABSENT {
            return false;
        }
        if entry.adds_key() {
            return self.transition_store_hit(table, entry, value);
        }
        let target = self.heap.tables.live_mut(table);
        if target.slot_value(entry.slot as usize).is_nil() {
            return false;
        }
        target.set_slot(entry.slot as usize, value);
        self.heap.table_barrier_back(table, value);
        true
    }

    /// Covers only a non-nil store into a non-black table with room and no metatable; everything else goes through `cached_store`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn transition_store_hit(&mut self, table: Ref<Table>, entry: Entry, value: Value) -> bool {
        let target = self.heap.tables.live(table);
        if value.is_nil() || target.metatable.is_some() || target.slots_full() || crate::gc::is_black(self.heap.tables.live_marks(table)) {
            return false;
        }
        let Some(marks) = self.heap.shapes.marks(entry.next) else {
            return false;
        };
        if marks & self.heap.gc.other_white() != 0 {
            return false;
        }
        let target = self.heap.tables.live_mut(table);
        target.enter_shape(entry.next);
        target.set_slot(entry.slot as usize, value);
        true
    }

    fn cached_transition(&mut self, proto: Ref<Proto>, site: usize, object: Value, value: Value) -> bool {
        let (Value::Table(table), Some(&InlineCache::Monomorphic(entry))) = (object, self.heap.protos.live(proto).exec.feedback.get(site)) else {
            return false;
        };
        entry.adds_key() && self.heap.tables.live(table).shape() == Some(entry.shape) && self.transition_store_hit(table, entry, value)
    }

    /// The site learns the method shape when the metatable chain is one `__index` hop.
    pub(super) fn absent_read(&mut self, proto: Ref<Proto>, site: usize, table: Ref<Table>, key: Value) -> Option<Value> {
        let Value::String(string) = key else {
            return None;
        };
        if let Some(value) = self.method_hit(proto, site, table) {
            return Some(value);
        }
        let value = self.index_chain_resolved(table, key);
        self.record_absent_chain(proto, site, table, string);
        value
    }

    /// The generic lookup covers what the cache does not, including metatable chains.
    pub(crate) fn read_field(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value) -> Option<Value> {
        let Some(string) = self.interned_key(key) else {
            return self.index_resolved(object, key);
        };
        if self.heap.protos.live(proto).exec.feedback.is_empty() {
            return self.index_resolved(object, key);
        }
        match self.cached_read(proto, site, object, string) {
            Cached::Value(value) => Some(value),
            Cached::Absent(table) => {
                if self.heap.tables.live(table).metatable.is_none() {
                    return Some(Value::Nil);
                }
                self.index_chain_resolved(table, key)
            }
            Cached::Unknown => self.index_resolved(object, key),
        }
    }

    /// False when the store may call `__newindex`, so it takes the generic path.
    pub(crate) fn store_field(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value, value: Value) -> bool {
        if self.cached_transition(proto, site, object, value) {
            return true;
        }
        let Some(string) = self.interned_key(key) else {
            return quick::raw_set_field(self, object, key, value);
        };
        if self.heap.protos.live(proto).exec.feedback.is_empty() {
            return quick::raw_set_field(self, object, key, value);
        }
        if self.cached_store(proto, site, object, value) {
            return true;
        }
        let before = self.shape_of_value(object);
        if !quick::raw_set_field(self, object, key, value) {
            return false;
        }
        self.record_store(proto, site, object, string, before);
        true
    }

    /// `None` leaves the read to the generic `index`.
    pub(super) fn field_read_probed(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value, probe: Probe) -> Option<Value> {
        match probe {
            Probe::Value(value) => Some(value),
            Probe::Absent(table) if self.absent_is_nil(table) => Some(Value::Nil),
            Probe::Absent(table) => self.absent_read(proto, site, table, key),
            Probe::Unknown => self.field_read_unprobed(proto, site, object, key),
        }
    }

    pub(super) fn field_read_unprobed(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value) -> Option<Value> {
        if self.heap.table_is_shaped(object) {
            return self.read_field(proto, site, object, key);
        }
        self.index_resolved(object, key)
    }

    pub(super) fn method_read_probed(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value, probe: Probe) -> Option<Value> {
        if let Probe::Value(method) = probe {
            return Some(method);
        }
        if let Value::Table(table) = object
            && let Some(method) = self.method_hit(proto, site, table)
        {
            return Some(method);
        }
        match (probe, self.site_feedback(proto, site)) {
            (Probe::Absent(table), _) => self.absent_read(proto, site, table, key),
            (_, Some(InlineCache::Method(methods))) => {
                let room = methods.len() < WAYS;
                self.method_miss(proto, site, object, key, room)
            }
            _ => self.read_field(proto, site, object, key),
        }
    }

    fn method_miss(&mut self, proto: Ref<Proto>, site: usize, object: Value, key: Value, room: bool) -> Option<Value> {
        let value = self.index_resolved(object, key);
        if room
            && let (Value::Table(table), Value::String(string)) = (object, key)
            && let Some(shape) = self.heap.tables.live(table).shape()
            && self.slot_in_shape(shape, string).is_none()
        {
            self.record_absent_chain(proto, site, table, string);
        }
        value
    }

    fn interned_key(&self, key: Value) -> Option<Ref<LuaString>> {
        match key {
            Value::String(string) if self.heap.string_keys[string.index()].interned => Some(string),
            _ => None,
        }
    }
}
