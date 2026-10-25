use std::ops::Range;


use crate::{
    arena::{Arena, Pinned, Ref, Slots},
    gc::{self, Gc, GcState},
    object::{Captured, Closure, LuaString, Native, StringBytes, Thread, UpState, UpVal, Userdata},
    proto::Proto,
    shape::{KeyOrigin, MAX_KEYS, MAX_TRANSITIONS, Shape, ShapeKey},
    table::{SetError, Table},
    value::Value,
};

pub const SHORT_STRING_LIMIT: usize = 40;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct StringKey {
    pub(crate) hash: u64,
    next: u32,
    pub(crate) interned: bool,
}

/// Follows `strt` in lstring.c.
#[derive(Clone, Default)]
struct StringTable {
    buckets: Vec<u32>,
    used: usize,
}

/// `MINSTRTABSIZE` in lstring.c.
const MIN_STRING_TABLE: usize = 128;

#[derive(Clone, Copy)]
pub struct KeyView<'a> {
    arena: &'a Arena<LuaString>,
    keys: &'a [StringKey],
    pub(crate) shapes: &'a Arena<Shape>,
}

impl<'a> KeyView<'a> {
    pub(crate) fn new(arena: &'a Arena<LuaString>, keys: &'a [StringKey], shapes: &'a Arena<Shape>) -> Self {
        Self { arena, keys, shapes }
    }

    pub(crate) fn hash(self, string: Ref<LuaString>) -> u64 {
        self.keys[string.index()].hash
    }

    pub(crate) fn is_interned(self, string: Ref<LuaString>) -> bool {
        self.keys[string.index()].interned
    }

    pub(crate) fn get(self, string: Ref<LuaString>) -> &'a LuaString {
        self.arena.get(string)
    }

    pub(crate) fn same_content(self, left: Ref<LuaString>, right: Ref<LuaString>) -> bool {
        let (left_key, right_key) = (self.keys[left.index()], self.keys[right.index()]);
        if (left_key.interned && right_key.interned) || left_key.hash != right_key.hash {
            return false;
        }
        match (self.arena.try_get(left), self.arena.try_get(right)) {
            (Some(left), Some(right)) => left.bytes == right.bytes,
            _ => false,
        }
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Heap {
    pub strings: Arena<LuaString>,
    pub(crate) string_keys: Vec<StringKey>,
    #[serde(skip)]
    short_strings: StringTable,
    pub(crate) events: Vec<Ref<LuaString>>,
    pub tables: Arena<Table>,
    pub(crate) shapes: Arena<Shape>,
    pub closures: Arena<Closure>,
    pub natives: Arena<Native>,
    pub upvalues: Arena<UpVal, Pinned<UpVal>>,
    pub protos: Arena<Proto>,
    pub threads: Arena<Thread>,
    pub userdata: Arena<Userdata>,
    pub(crate) gc: GcState,
    #[serde(skip)]
    pub(crate) white: Vec<Gc>,
    #[serde(skip)]
    pub(crate) compiler: crate::compiler::Scratch,
}

pub(crate) trait Collectable: Sized + Default {
    type Slots: Slots<Self>;
    fn arena(heap: &mut Heap) -> &mut Arena<Self, Self::Slots>;
    fn wrap(reference: Ref<Self>) -> Gc;
    fn size(&self) -> usize;
}

macro_rules! collectable {
    ($ty:ty, $field:ident, $variant:ident, $object:ident => $size:expr) => {
        collectable!($ty, Vec<$ty>, $field, $variant, $object => $size);
    };
    ($ty:ty, $slots:ty, $field:ident, $variant:ident, $object:ident => $size:expr) => {
        impl Collectable for $ty {
            type Slots = $slots;

            fn arena(heap: &mut Heap) -> &mut Arena<Self, Self::Slots> {
                &mut heap.$field
            }

            fn wrap(reference: Ref<Self>) -> Gc {
                Gc::$variant(reference)
            }

            fn size(&self) -> usize {
                let $object = self;
                $size
            }
        }
    };
}

collectable!(LuaString, strings, String, object => 40 + object.bytes.len());
collectable!(Table, tables, Table, object => object.size());
collectable!(Closure, closures, Closure, object => 32 + 8 * object.upvalues.len());
collectable!(Native, natives, Native, object => 48 + 16 * object.upvalues.len());
collectable!(UpVal, Pinned<UpVal>, upvalues, UpVal, _object => 40);
collectable!(Proto, protos, Proto, object => 128 + 4 * object.code.len() + 16 * object.constants.len());
collectable!(Thread, threads, Thread, object => object.size());
collectable!(Userdata, userdata, Userdata, object => 48 + 16 * object.user_values.len());
collectable!(Shape, shapes, Shape, object => object.size());

impl Heap {
    #[inline(always)]
    pub(crate) fn alloc<T: Collectable>(&mut self, object: T) -> Ref<T> {
        let size = object.size();
        let white = self.gc.current_white;
        let reference = T::arena(self).insert_marked(object, white);
        self.gc.grow(size as i64);
        if self.gc.is_generational() {
            self.gc.young.push(T::wrap(reference));
        }
        reference
    }

    pub fn string(&mut self, bytes: &[u8]) -> Ref<LuaString> {
        if bytes.len() > SHORT_STRING_LIMIT {
            return self.new_string(StringBytes::from_slice(bytes), hash_bytes(bytes), false);
        }
        let hash = hash_bytes(bytes);
        if let Some(index) = self.find_short_string(bytes, hash) {
            self.revive_dead_string_at(index);
            return self.strings.reference_at(index);
        }
        if self.short_strings.used >= self.short_strings.buckets.len() {
            self.grow_short_strings();
        }
        let reference = self.new_string(StringBytes::from_slice(bytes), hash, true);
        self.link_short_string(reference.index());
        reference
    }

    #[inline]
    pub(crate) fn table_is_shaped(&self, value: Value) -> bool {
        match value {
            Value::Table(table) => self.tables.live(table).shape().is_some(),
            _ => false,
        }
    }

    #[inline]
    pub(crate) fn key_view(&self) -> KeyView<'_> {
        KeyView::new(&self.strings, &self.string_keys, &self.shapes)
    }

    pub(crate) fn rebuild_short_strings(&mut self) {
        self.short_strings = StringTable::default();
        let interned: Vec<usize> = self
            .strings
            .iter()
            .map(|(reference, _)| reference.index())
            .filter(|&index| self.string_keys[index].interned)
            .collect();
        for index in interned {
            if self.short_strings.used >= self.short_strings.buckets.len() {
                self.grow_short_strings();
            }
            self.link_short_string(index);
        }
    }

    /// Follows `internshrstr` in lstring.c.
    fn find_short_string(&self, bytes: &[u8], hash: u64) -> Option<usize> {
        let buckets = &self.short_strings.buckets;
        if buckets.is_empty() {
            return None;
        }
        let mut link = buckets[hash as usize & (buckets.len() - 1)];
        while link != 0 {
            let index = link as usize - 1;
            let stored = self.strings.object_at(index).bytes.as_slice();
            if stored.len() == bytes.len() && stored == bytes {
                return Some(index);
            }
            link = self.string_keys[index].next;
        }
        None
    }

    fn link_short_string(&mut self, index: usize) {
        let table = &mut self.short_strings;
        let bucket = self.string_keys[index].hash as usize & (table.buckets.len() - 1);
        self.string_keys[index].next = table.buckets[bucket];
        table.buckets[bucket] = index as u32 + 1;
        table.used += 1;
    }

    /// Follows `growstrtab` in lstring.c.
    fn grow_short_strings(&mut self) {
        let size = (self.short_strings.buckets.len() * 2).max(MIN_STRING_TABLE);
        self.resize_short_strings(size);
    }

    /// Follows `checkSizes` in lgc.c.
    pub(crate) fn check_string_table_size(&mut self) {
        let size = self.short_strings.buckets.len();
        if self.short_strings.used < size / 4 {
            self.resize_short_strings(size / 2);
        }
    }

    /// Follows `luaS_resize` and `tablerehash` in lstring.c.
    fn resize_short_strings(&mut self, size: usize) {
        debug_assert!(size.is_power_of_two());
        let old = std::mem::replace(&mut self.short_strings.buckets, vec![0; size]);
        for head in old {
            let mut link = head;
            while link != 0 {
                let index = link as usize - 1;
                let key = &mut self.string_keys[index];
                link = key.next;
                let bucket = key.hash as usize & (size - 1);
                key.next = self.short_strings.buckets[bucket];
                self.short_strings.buckets[bucket] = index as u32 + 1;
            }
        }
    }

    /// Interning a dead, unswept string revives it, as `internshrstr` runs `changewhite`.
    fn revive_dead_string_at(&mut self, index: usize) {
        let (other_white, white) = (self.gc.other_white(), self.gc.current_white);
        let marks = self.strings.marks_at_mut(index);
        if *marks & other_white != 0 {
            *marks = (*marks & !gc::GC_BITS) | white;
        }
    }

    /// A table entering a dead shape makes it live, so the sweep keeps it.
    pub(crate) fn revive_dead_shape(&mut self, shape: Ref<Shape>) {
        let Some(marks) = self.shapes.marks(shape) else {
            return;
        };
        if marks & self.gc.other_white() == 0 {
            return;
        }
        let white = self.gc.current_white;
        self.shapes.set_marks(shape, (marks & !gc::GC_BITS) | white);
    }

    /// Follows `luaS_remove` in lstring.c.
    pub(crate) fn forget_short_string(&mut self, reference: Ref<LuaString>) {
        let index = reference.index();
        let target = index as u32 + 1;
        let after = self.string_keys[index].next;
        let buckets = &mut self.short_strings.buckets;
        let bucket = self.string_keys[index].hash as usize & (buckets.len() - 1);
        self.short_strings.used -= 1;
        if buckets[bucket] == target {
            buckets[bucket] = after;
            return;
        }
        let mut link = buckets[bucket];
        loop {
            debug_assert!(link != 0, "an interned string is in its bucket's chain");
            let previous = link as usize - 1;
            link = self.string_keys[previous].next;
            if link == target {
                self.string_keys[previous].next = after;
                return;
            }
        }
    }

    pub fn string_from_buffer(&mut self, bytes: Vec<u8>) -> Ref<LuaString> {
        if bytes.len() > SHORT_STRING_LIMIT {
            let hash = hash_bytes(&bytes);
            return self.new_string(StringBytes::from_vec(bytes), hash, false);
        }
        self.string(&bytes)
    }

    fn new_string(&mut self, bytes: StringBytes, hash: u64, interned: bool) -> Ref<LuaString> {
        let reference = self.alloc(LuaString { bytes });
        let index = reference.index();
        if self.string_keys.len() <= index {
            self.string_keys.resize(index + 1, StringKey::default());
        }
        self.string_keys[index] = StringKey { hash, next: 0, interned };
        reference
    }

    /// Neither is stored in a snapshot, so restored instructions start uncached.
    pub(crate) fn rebuild_feedback(&mut self) {
        for proto in self.protos.iter_mut() {
            *proto.exec = crate::vm::Exec::new(&proto.code);
        }
    }

    pub fn bytes(&self, string: Ref<LuaString>) -> &[u8] {
        &self.strings.get(string).bytes
    }

    pub fn string_value(&mut self, bytes: &[u8]) -> Value {
        Value::String(self.string(bytes))
    }

    pub fn raw_equal(&self, left: Value, right: Value) -> bool {
        match (left, right) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Integer(i), Value::Float(f)) | (Value::Float(f), Value::Integer(i)) => {
                crate::number::float_equals_integer(f, i)
            }
            (Value::String(a), Value::String(b)) => self.strings_equal(a, b),
            (Value::Table(a), Value::Table(b)) => a == b,
            (Value::Closure(a), Value::Closure(b)) => a == b,
            (Value::Native(a), Value::Native(b)) => a == b,
            (Value::Thread(a), Value::Thread(b)) => a == b,
            (Value::Userdata(a), Value::Userdata(b)) => a == b,
            (Value::LightUserdata(a), Value::LightUserdata(b)) => a == b,
            _ => false,
        }
    }

    pub fn strings_equal(&self, a: Ref<LuaString>, b: Ref<LuaString>) -> bool {
        a == b || self.key_view().same_content(a, b)
    }

    #[inline]
    pub(crate) fn marks_of(&self, gc: Gc) -> Option<u8> {
        match gc {
            Gc::String(reference) => self.strings.marks(reference),
            Gc::Table(reference) => self.tables.marks(reference),
            Gc::Closure(reference) => self.closures.marks(reference),
            Gc::Native(reference) => self.natives.marks(reference),
            Gc::UpVal(reference) => self.upvalues.marks(reference),
            Gc::Proto(reference) => self.protos.marks(reference),
            Gc::Thread(reference) => self.threads.marks(reference),
            Gc::Userdata(reference) => self.userdata.marks(reference),
            Gc::Shape(reference) => self.shapes.marks(reference),
        }
    }

    #[inline]
    pub(crate) fn set_marks_of(&mut self, gc: Gc, marks: u8) {
        match gc {
            Gc::String(reference) => self.strings.set_marks(reference, marks),
            Gc::Table(reference) => self.tables.set_marks(reference, marks),
            Gc::Closure(reference) => self.closures.set_marks(reference, marks),
            Gc::Native(reference) => self.natives.set_marks(reference, marks),
            Gc::UpVal(reference) => self.upvalues.set_marks(reference, marks),
            Gc::Proto(reference) => self.protos.set_marks(reference, marks),
            Gc::Thread(reference) => self.threads.set_marks(reference, marks),
            Gc::Userdata(reference) => self.userdata.set_marks(reference, marks),
            Gc::Shape(reference) => self.shapes.set_marks(reference, marks),
        }
    }

    #[inline]
    pub(crate) fn object_size(&self, gc: Gc) -> i64 {
        let size = match gc {
            Gc::String(reference) => self.strings.try_get(reference).map(Collectable::size),
            Gc::Table(reference) => self.tables.try_get(reference).map(Collectable::size),
            Gc::Closure(reference) => self.closures.try_get(reference).map(Collectable::size),
            Gc::Native(reference) => self.natives.try_get(reference).map(Collectable::size),
            Gc::UpVal(reference) => self.upvalues.try_get(reference).map(Collectable::size),
            Gc::Proto(reference) => self.protos.try_get(reference).map(Collectable::size),
            Gc::Thread(reference) => self.threads.try_get(reference).map(Collectable::size),
            Gc::Userdata(reference) => self.userdata.try_get(reference).map(Collectable::size),
            Gc::Shape(reference) => self.shapes.try_get(reference).map(Collectable::size),
        };
        size.map_or(0, |bytes| bytes as i64)
    }

    #[inline]
    pub(crate) fn set_age(&mut self, gc: Gc, age: u8) {
        if let Some(marks) = self.marks_of(gc) {
            self.set_marks_of(gc, gc::with_age(marks, age));
        }
    }

    pub(crate) fn barrier(&mut self, parent: Gc, child: Option<Gc>) {
        let Some(child) = child else {
            return;
        };
        let Some(parent_marks) = self.marks_of(parent).filter(|&marks| gc::is_black(marks)) else {
            return;
        };
        if !self.marks_of(child).is_some_and(gc::is_white) {
            return;
        }
        if self.gc.keeps_invariant() {
            self.really_mark(child);
            if gc::is_old(parent_marks) {
                self.set_age(child, gc::OLD0);
            }
        } else if !self.gc.is_generational() {
            let white = self.gc.current_white;
            self.set_marks_of(parent, gc::make_white(parent_marks, white));
        }
    }

    pub(crate) fn barrier_back(&mut self, parent: Gc) {
        let Some(marks) = self.marks_of(parent) else {
            return;
        };
        if gc::age(marks) == gc::TOUCHED2 {
            self.set_marks_of(parent, gc::set_gray(marks));
        } else {
            self.link_gray(parent, GrayList::Again);
        }
        if gc::is_old(marks) {
            self.set_age(parent, gc::TOUCHED1);
        }
    }

    pub(crate) fn barrier_value(&mut self, parent: Gc, value: Value) {
        self.barrier(parent, gc::value_gc(value));
    }

    #[inline]
    pub(crate) fn barrier_back_for(&mut self, parent: Gc, key: Value, value: Value) {
        let Some(marks) = self.marks_of(parent) else {
            return;
        };
        if !gc::is_black(marks) {
            return;
        }
        let white = |heap: &Heap, value: Value| {
            gc::value_gc(value).is_some_and(|child| heap.marks_of(child).is_none_or(gc::is_white))
        };
        if white(self, key) || white(self, value) {
            self.barrier_back(parent);
        }
    }

    /// Returns false when the shape is gone; a shape the sweep has not reached is revived.
    pub(crate) fn take_transition(&mut self, table: Ref<Table>, next: Ref<Shape>) -> bool {
        if self.shapes.marks(next).is_none() {
            return false;
        }
        self.revive_dead_shape(next);
        if gc::is_black(self.tables.live_marks(table)) {
            self.barrier(Gc::Table(table), Some(Gc::Shape(next)));
        }
        let grows = self.tables.live(table).slots_full();
        let before = grows.then(|| self.tables.live(table).size());
        self.tables.live_mut(table).enter_shape(next);
        if let Some(before) = before {
            let after = self.tables.live(table).size();
            self.grow_by(after, before);
        }
        true
    }

    /// Follows `luaC_barrierback` in lgc.c.
    #[inline]
    pub(crate) fn table_barrier_back(&mut self, table: Ref<Table>, value: Value) {
        if !value.is_collectable() {
            return;
        }
        let Some(child) = gc::value_gc(value) else {
            return;
        };
        if !gc::is_black(self.tables.live_marks(table)) {
            return;
        }
        if self.marks_of(child).is_none_or(gc::is_white) {
            self.barrier_back(Gc::Table(table));
        }
    }

    #[inline]
    pub(crate) fn table_set(&mut self, table: Ref<Table>, key: Value, value: Value) -> Result<(), SetError> {
        self.table_store(table, key, value, KeyOrigin::Dynamic)
    }

    #[inline]
    pub(crate) fn table_set_field(&mut self, table: Ref<Table>, key: Value, value: Value) -> Result<(), SetError> {
        self.table_store(table, key, value, KeyOrigin::Constant)
    }

    #[inline]
    pub(crate) fn table_store(&mut self, table: Ref<Table>, key: Value, value: Value, origin: KeyOrigin) -> Result<(), SetError> {
        if let Value::Integer(index) = key
            && self.tables.live_mut(table).replace_int(index, &value)
        {
            self.table_barrier_back(table, value);
            return Ok(());
        }
        if let Value::String(string) = key
            && self.string_keys[string.index()].interned
        {
            let keys = KeyView::new(&self.strings, &self.string_keys, &self.shapes);
            if self.tables.live_mut(table).replace_present_string(keys, string, value) {
                self.table_barrier_back(table, value);
                return Ok(());
            }
        }
        self.table_set_sized(table, key, value, origin)
    }

    #[inline(never)]
    fn table_set_sized(&mut self, table: Ref<Table>, key: Value, value: Value, origin: KeyOrigin) -> Result<(), SetError> {
        let before = self.tables.live(table).size();
        let headroom = self.headroom();
        let taken = self.add_shape_key(table, key, value, origin, headroom)?;
        let headroom = headroom.map(|room| room.saturating_sub(taken));
        let keys = KeyView::new(&self.strings, &self.string_keys, &self.shapes);
        let outcome = self.tables.live_mut(table).set_within(keys, key, value, headroom);
        let after = self.tables.live(table).size();
        self.grow_by(after, before);
        if outcome.is_ok() {
            self.barrier_back_for(Gc::Table(table), key, value);
        }
        outcome
    }

    /// When the transition tree refuses the key, the table leaves shape mode instead.
    #[inline(never)]
    fn add_shape_key(&mut self, table: Ref<Table>, key: Value, value: Value, origin: KeyOrigin, headroom: Option<usize>) -> Result<usize, SetError> {
        let Some(shape) = self.tables.live(table).shape() else {
            return Ok(0);
        };
        let Value::String(string) = key else {
            return Ok(0);
        };
        if value.is_nil() || !self.string_keys[string.index()].interned {
            return Ok(0);
        }
        let hash = self.string_keys[string.index()].hash;
        if self.shapes.live(shape).slot(&self.shapes, string, hash).is_some() {
            return Ok(0);
        }
        let Some(child) = self.shape_for_new_key(table, shape, ShapeKey { string, hash }, origin) else {
            self.leave_table_shape(table, headroom)?;
            return Ok(0);
        };
        let taken = self.tables.live_mut(table).reserve_slot(headroom)?;
        self.barrier(Gc::Table(table), Some(Gc::Shape(child)));
        self.tables.live_mut(table).enter_shape(child);
        Ok(taken)
    }

    /// Returns `None` when a limit refuses a new child; an existing child is always reused.
    fn shape_for_new_key(&mut self, table: Ref<Table>, shape: Ref<Shape>, key: ShapeKey, origin: KeyOrigin) -> Option<Ref<Shape>> {
        if let Some(child) = self.existing_shape_child(shape, key) {
            self.revive_dead_shape(child);
            return Some(child);
        }
        let parent = self.shapes.live(shape);
        let full = parent.len() >= MAX_KEYS;
        let fanned_out = origin == KeyOrigin::Dynamic && parent.transition_count() >= MAX_TRANSITIONS;
        let budget_spent = origin == KeyOrigin::Dynamic && !self.tables.live(table).may_create_dynamic_transition();
        if full || fanned_out || budget_spent {
            return None;
        }
        if origin == KeyOrigin::Dynamic {
            self.tables.live_mut(table).count_dynamic_transition();
        }
        Some(self.create_shape_child(shape, key))
    }

    fn leave_table_shape(&mut self, table: Ref<Table>, headroom: Option<usize>) -> Result<(), SetError> {
        let keys = KeyView::new(&self.strings, &self.string_keys, &self.shapes);
        self.tables.live_mut(table).leave_shape(keys, headroom)
    }

    pub(crate) fn table_set_array_from_thread(
        &mut self,
        table: Ref<Table>,
        first: usize,
        thread: Ref<Thread>,
        slots: Range<usize>,
    ) -> Result<(), SetError> {
        let before = self.tables.get(table).size();
        let headroom = self.headroom();
        let values = &self.threads.live(thread).stack[slots.clone()];
        self.tables.get_mut(table).set_array_items(first, values, headroom)?;
        let after = self.tables.get(table).size();
        self.grow_by(after, before);
        if !gc::is_black(self.tables.live_marks(table)) {
            return Ok(());
        }
        for slot in slots {
            let value = self.threads.live(thread).stack[slot];
            self.barrier_value(Gc::Table(table), value);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn headroom(&self) -> Option<usize> {
        self.gc.limit.map(|limit| (limit as i128 - i128::from(self.gc.total)).max(0) as usize)
    }

    pub(crate) fn set_table_metatable(&mut self, table: Ref<Table>, metatable: Option<Ref<Table>>) {
        self.tables.get_mut(table).metatable = metatable;
        self.barrier(Gc::Table(table), metatable.map(Gc::Table));
    }

    pub(crate) fn set_userdata_metatable(&mut self, userdata: Ref<Userdata>, metatable: Option<Ref<Table>>) {
        self.userdata.get_mut(userdata).metatable = metatable;
        self.barrier(Gc::Userdata(userdata), metatable.map(Gc::Table));
    }

    pub(crate) fn set_userdata_value(&mut self, userdata: Ref<Userdata>, index: usize, value: Value) {
        self.userdata.get_mut(userdata).user_values[index] = value;
        self.barrier_value(Gc::Userdata(userdata), value);
    }

    pub(crate) fn set_native_value(&mut self, native: Ref<Native>, index: usize, value: Value) {
        self.natives.get_mut(native).upvalues[index] = value;
        self.barrier_value(Gc::Native(native), value);
    }

    pub(crate) fn set_closure_upvalue(&mut self, closure: Ref<Closure>, index: usize, upvalue: Ref<UpVal>) {
        let captured = self.captured(upvalue);
        self.closures.get_mut(closure).upvalues[index] = captured;
        self.barrier(Gc::Closure(closure), Some(Gc::UpVal(upvalue)));
    }

    pub(crate) fn captured(&mut self, upvalue: Ref<UpVal>) -> Captured {
        Captured {
            reference: upvalue,
            cell: crate::vm::Raw::new(self.upvalues.address(upvalue)),
        }
    }

    pub(crate) fn alloc_open_upvalue(&mut self, thread: Ref<Thread>, index: usize) -> Ref<UpVal> {
        let slot = self.threads.live_mut(thread).stack.as_mut_ptr().wrapping_add(index);
        self.alloc(UpVal::open(thread, index, slot))
    }

    pub(crate) fn alloc_closed_upvalue(&mut self, value: Value) -> Ref<UpVal> {
        let upvalue = self.alloc(UpVal::default());
        self.upvalues.live_mut(upvalue).close(value);
        upvalue
    }

    /// Follows `correctstack` in ldo.c.
    pub(crate) fn stack_resized(&mut self, thread: Ref<Thread>) {
        let state = self.threads.live_mut(thread);
        state.stack_resized();
        let stack = state.stack.as_mut_ptr();
        for &upvalue in &state.open_upvalues {
            let cell = self.upvalues.live_mut(upvalue);
            let UpState::Open { index, .. } = cell.state else {
                unreachable!("the open list holds open upvalues");
            };
            cell.point_at(stack.wrapping_add(index));
        }
    }

    /// A snapshot does not store these pointers, so restore fills them.
    pub(crate) fn rebuild_upvalues(&mut self) {
        let upvalues: Vec<Ref<UpVal>> = self.upvalues.iter().map(|(reference, _)| reference).collect();
        for upvalue in upvalues {
            let cell = self.upvalues.live_mut(upvalue);
            if let UpState::Closed(value) = cell.state {
                cell.close(value);
            }
        }
        let threads: Vec<Ref<Thread>> = self.threads.iter().map(|(reference, _)| reference).collect();
        for thread in threads {
            self.stack_resized(thread);
        }
        let closures: Vec<Ref<Closure>> = self.closures.iter().map(|(reference, _)| reference).collect();
        for closure in closures {
            for index in 0..self.closures.live(closure).upvalues.len() {
                let upvalue = self.closures.live(closure).upvalues[index].reference;
                let cell = crate::vm::Raw::new(self.upvalues.address(upvalue));
                self.closures.live_mut(closure).upvalues[index].cell = cell;
            }
        }
    }

    /// Follows `luaF_closeupval` in lfunc.c; a white upvalue needs no barrier.
    pub(crate) fn set_closed_upvalue(&mut self, upvalue: Ref<UpVal>, value: Value) {
        self.upvalues.live_mut(upvalue).close(value);
        self.closed_upvalue_barrier(upvalue, value);
    }

    pub(crate) fn closed_upvalue_barrier(&mut self, upvalue: Ref<UpVal>, value: Value) {
        let marks = self.upvalues.live_marks(upvalue);
        if gc::is_white(marks) {
            return;
        }
        self.upvalues.set_marks(upvalue, gc::set_black(marks));
        self.barrier_value(Gc::UpVal(upvalue), value);
    }

    /// Follows `luaF_closeupval` in lfunc.c, closing from the top of the open list.
    pub(crate) fn close_upvalues(&mut self, thread: Ref<Thread>, level: usize) {
        loop {
            let state = self.threads.live_mut(thread);
            let Some(&upvalue) = state.open_upvalues.last() else {
                return;
            };
            let UpState::Open { index, .. } = self.upvalues.live(upvalue).state else {
                unreachable!("the open list holds open upvalues");
            };
            if index < level {
                return;
            }
            let value = state.stack[index];
            state.open_upvalues.pop();
            self.set_closed_upvalue(upvalue, value);
        }
    }

    pub(crate) fn grow_by(&mut self, after: usize, before: usize) {
        self.gc.grow(after as i64 - before as i64);
    }
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum GrayList {
    Gray,
    Again,
    Weak,
    AllWeak,
    Ephemeron,
}

/// Follows `luaS_hash` in lstring.c with seed 0, the seed the harness builds PUC with.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let hash = bytes.iter().rev().fold(bytes.len() as u32, |hash, &byte| {
        hash ^ (hash << 5).wrapping_add(hash >> 2).wrapping_add(u32::from(byte))
    });
    u64::from(hash)
}
