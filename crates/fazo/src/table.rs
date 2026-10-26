use std::cell::Cell;

use crate::{
    arena::Ref,
    heap::KeyView,
    number,
    object::LuaString,
    shape::{MAX_DYNAMIC_TRANSITIONS, ROOT, Shape, ShapeKey},
    value::Value,
};

const MAX_ARRAY_BITS: usize = 31;
const MAX_VICINITY: usize = 4;
const HASH_SEARCH_SEED: u64 = 0x9E37_79B9;
const SMALL_ARRAY_BYTES: usize = 1024;

#[repr(C)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub metatable: Option<Ref<Table>>,
    absent: u8,
    shaped: bool,
    shape: Ref<Shape>,
    dynamic_transitions: u8,
    array: Vec<Value>,
    slots: Vec<Value>,
    nodes: Vec<Node>,
    last_free: usize,
    length_hint: Cell<usize>,
}

/// Compiled code reads these fields at fixed byte offsets.
#[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod offsets {
    use super::Table;

    pub(crate) const METATABLE: usize = std::mem::offset_of!(Table, metatable);
    pub(crate) const ABSENT: usize = std::mem::offset_of!(Table, absent);
    pub(crate) const SHAPED: usize = std::mem::offset_of!(Table, shaped);
    pub(crate) const SHAPE: usize = std::mem::offset_of!(Table, shape);
    pub(crate) const SLOTS: usize = std::mem::offset_of!(Table, slots);
    pub(crate) const ARRAY: usize = std::mem::offset_of!(Table, array);
    pub(crate) const NODES: usize = std::mem::offset_of!(Table, nodes);
    pub(crate) const NODE_SIZE: usize = std::mem::size_of::<super::Node>();
    pub(crate) const NODE_KEY: usize = std::mem::offset_of!(super::Node, key);
    pub(crate) const NODE_VALUE: usize = std::mem::offset_of!(super::Node, value);
    pub(crate) const NODE_NEXT: usize = std::mem::offset_of!(super::Node, next);
    pub(crate) const NODE_DEAD: usize = std::mem::offset_of!(super::Node, dead);
}

impl Default for Table {
    fn default() -> Self {
        Self {
            metatable: None,
            absent: 0,
            shaped: true,
            shape: ROOT,
            dynamic_transitions: 0,
            array: Vec::new(),
            slots: Vec::new(),
            nodes: Vec::new(),
            last_free: 0,
            length_hint: Cell::new(0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
struct Node {
    key: Value,
    value: Value,
    hash: u64,
    next: u32,
    dead: bool,
}

impl Node {
    const EMPTY: Self = Self {
        key: Value::Nil,
        value: Value::Nil,
        hash: 0,
        next: 0,
        dead: false,
    };
}

#[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SetError {
    NilKey,
    NanKey,
    Memory,
}

impl Table {
    pub fn sized_bytes(array: usize, hash: usize) -> usize {
        let nodes = if hash == 0 { 0 } else { hash.next_power_of_two() };
        array * std::mem::size_of::<Value>() + nodes * std::mem::size_of::<Node>()
    }

    pub fn with_sizes(array: usize, hash: usize) -> Result<Self, SetError> {
        Self::with_spare(array, 0, hash)
    }

    #[inline]
    pub(crate) fn with_spare(array: usize, spare: usize, hash: usize) -> Result<Self, SetError> {
        let slots = reserved_values(hash)?;
        let array_values = filled_values(array, spare)?;

        Ok(Self {
            metatable: None,
            absent: 0,
            shaped: true,
            shape: ROOT,
            dynamic_transitions: 0,
            array: array_values,
            slots,
            nodes: Vec::new(),
            last_free: 0,
            length_hint: Cell::new(array / 2),
        })
    }

    #[inline]
    pub fn get(&self, keys: KeyView, key: Value) -> Value {
        if let Value::Integer(index) = key {
            return self.get_int(index);
        }
        self.get_hashed(keys, key)
    }

    #[inline]
    fn get_hashed(&self, keys: KeyView, key: Value) -> Value {
        match number::normalize_key(key) {
            Value::Integer(index) => self.get_int(index),
            Value::String(string) if keys.is_interned(string) => self.get_short_str(keys, string, keys.hash(string)),
            key => self.get_node(keys, key),
        }
    }

    fn get_node(&self, keys: KeyView, key: Value) -> Value {
        if self.nodes.is_empty() {
            return Value::Nil;
        }
        let hash = key_hash(keys, key);
        self.find(keys, key, hash, false)
            .map_or(Value::Nil, |node| self.nodes[node].value)
    }

    #[inline(always)]
    pub(crate) fn get_short_str(&self, keys: KeyView, string: Ref<LuaString>, hash: u64) -> Value {
        if self.shaped {
            return self.slot_of(keys, string, hash).map_or(Value::Nil, |slot| self.slots[slot]);
        }
        *self.node_short_str(string, hash)
    }

    /// Follows `luaH_getshortstr` in ltable.c. A dead node's value is nil and dictionary mode always has a node, so the walk needs neither test.
    #[inline(always)]
    pub(crate) fn node_short_str(&self, string: Ref<LuaString>, hash: u64) -> &Value {
        static NIL: Value = Value::Nil;
        debug_assert!(!self.shaped && !self.nodes.is_empty());
        let mut index = hash as usize & (self.nodes.len() - 1);
        loop {
            let node = &self.nodes[index];
            debug_assert!(!node.dead || node.value.is_nil());
            if matches!(node.key, Value::String(stored) if stored == string) {
                return &node.value;
            }
            match node.next {
                0 => return &NIL,
                next => index = next as usize - 1,
            }
        }
    }

    /// Follows PUC's `luaV_fastseti`; a store that may add the key or resize goes through `set_within`.
    #[inline]
    pub(crate) fn replace_int(&mut self, index: i64, value: &Value) -> bool {
        if let Some(slot) = array_slot(index, self.array.len()) {
            self.array[slot] = *value;
            return true;
        }
        self.replace_present_node_int(index, *value)
    }

    /// Follows `luaH_fastseti` in ltable.h; an empty array item of a table with a metatable leaves `__newindex` to the caller.
    #[inline(always)]
    pub(crate) fn fast_set_int(&mut self, index: i64, value: &Value) -> bool {
        let Some(slot) = array_slot(index, self.array.len()) else {
            return self.replace_present_node_int(index, *value);
        };
        let place = &mut self.array[slot];
        if self.metatable.is_some() && place.is_nil() {
            return false;
        }
        *place = *value;
        true
    }

    /// A table without a metatable has no `__newindex` to run, so the store can skip it.
    #[inline(always)]
    pub(crate) fn replace_array_item(&mut self, index: i64, value: &Value) -> bool {
        let Some(slot) = array_slot(index, self.array.len()) else {
            return false;
        };
        let place = &mut self.array[slot];
        if self.metatable.is_some() && place.is_nil() {
            return false;
        }
        *place = *value;
        true
    }

    #[inline(never)]
    fn replace_present_node_int(&mut self, index: i64, value: Value) -> bool {
        let Some(node) = self.find_int(index) else {
            return false;
        };
        let place = &mut self.nodes[node].value;
        if place.is_nil() {
            return false;
        }
        *place = value;
        true
    }

    #[inline]
    pub fn get_int(&self, index: i64) -> Value {
        if let Some(slot) = array_slot(index, self.array.len()) {
            return self.array[slot];
        }
        self.get_node_int(index)
    }

    #[inline]
    pub(crate) fn int_ref(&self, index: i64) -> &Value {
        if let Some(slot) = array_slot(index, self.array.len()) {
            return &self.array[slot];
        }
        self.node_int_ref(index)
    }

    #[inline(never)]
    fn node_int_ref(&self, index: i64) -> &Value {
        static NIL: Value = Value::Nil;
        self.find_int(index).map_or(&NIL, |node| &self.nodes[node].value)
    }

    #[inline(never)]
    fn get_node_int(&self, index: i64) -> Value {
        self.find_int(index)
            .map_or(Value::Nil, |node| self.nodes[node].value)
    }

    fn find_int(&self, index: i64) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut position = self.main_position(Value::Integer(index), index as u64);
        loop {
            let node = &self.nodes[position];
            if matches!(node.key, Value::Integer(key) if key == index) && !node.dead {
                return Some(position);
            }
            match node.next {
                0 => return None,
                next => position = next as usize - 1,
            }
        }
    }

    /// Adding a key to a shape is `Heap::table_set`'s job; this method leaves the shape only when that step was skipped.
    #[inline]
    pub fn set(&mut self, keys: KeyView, key: Value, value: Value) -> Result<(), SetError> {
        self.set_within(keys, key, value, None)
    }

    /// A growth that does not fit or cannot be allocated returns `SetError::Memory` and changes nothing.
    #[inline]
    pub(crate) fn set_within(&mut self, keys: KeyView, key: Value, value: Value, headroom: Option<usize>) -> Result<(), SetError> {
        if let Value::Integer(index) = key
            && let Some(slot) = array_slot(index, self.array.len())
        {
            self.array[slot] = value;
            return Ok(());
        }
        self.set_hashed(keys, key, value, headroom)
    }

    #[inline(never)]
    fn set_hashed(&mut self, keys: KeyView, key: Value, value: Value, headroom: Option<usize>) -> Result<(), SetError> {
        let key = match number::normalize_key(key) {
            Value::Nil => return Err(SetError::NilKey),
            Value::Float(f) if f.is_nan() => return Err(SetError::NanKey),
            key => key,
        };
        if let Value::Integer(index) = key
            && let Some(slot) = array_slot(index, self.array.len())
        {
            self.array[slot] = value;
            return Ok(());
        }
        if !value.is_nil() {
            self.forget_absence(keys, key);
        }
        if let Value::String(string) = key
            && self.shaped
            && keys.is_interned(string)
        {
            if let Some(slot) = self.slot_of(keys, string, keys.hash(string)) {
                self.slots[slot] = value;
                return Ok(());
            }
            if value.is_nil() {
                return Ok(());
            }
            self.leave_shape(keys, headroom)?;
        }
        let hash = key_hash(keys, key);
        if let Some(node) = self.find(keys, key, hash, false) {
            self.nodes[node].value = value;
            return Ok(());
        }
        if !value.is_nil() {
            self.insert_new(key, hash, value, headroom)?;
        }
        Ok(())
    }

    /// Follows PUC's `luaV_fastset`.
    #[inline]
    pub(crate) fn replace_present_string(&mut self, keys: KeyView, string: Ref<LuaString>, value: Value) -> bool {
        let hash = keys.hash(string);
        let place = if self.shaped {
            self.slot_of(keys, string, hash).map(|slot| &mut self.slots[slot])
        } else {
            self.node_value_short_str(string, hash)
        };
        let Some(place) = place.filter(|place| !place.is_nil()) else {
            return false;
        };
        *place = value;
        true
    }

    fn node_value_short_str(&mut self, string: Ref<LuaString>, hash: u64) -> Option<&mut Value> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut index = hash as usize & (self.nodes.len() - 1);
        loop {
            let node = &self.nodes[index];
            if node.hash == hash && !node.dead && matches!(node.key, Value::String(stored) if stored == string) {
                return Some(&mut self.nodes[index].value);
            }
            match node.next {
                0 => return None,
                next => index = next as usize - 1,
            }
        }
    }

    #[inline]
    fn slot_of(&self, keys: KeyView, string: Ref<LuaString>, hash: u64) -> Option<usize> {
        keys.shapes.live(self.shape).slot(keys.shapes, string, hash)
    }

    #[inline]
    pub(crate) fn shape(&self) -> Option<Ref<Shape>> {
        self.shaped.then_some(self.shape)
    }

    #[inline]
    pub(crate) fn slots_full(&self) -> bool {
        self.slots.len() == self.slots.capacity()
    }

    #[inline]
    pub(crate) fn enter_shape(&mut self, shape: Ref<Shape>) {
        self.shape = shape;
        self.slots.push(Value::Nil);
    }

    pub(crate) fn may_create_dynamic_transition(&self) -> bool {
        self.dynamic_transitions < MAX_DYNAMIC_TRANSITIONS
    }

    pub(crate) fn count_dynamic_transition(&mut self) {
        self.dynamic_transitions += 1;
    }

    pub(crate) fn leave_shape(&mut self, keys: KeyView, headroom: Option<usize>) -> Result<(), SetError> {
        let entries: Vec<(ShapeKey, Value)> = keys
            .shapes
            .live(self.shape)
            .keys(keys.shapes)
            .into_iter()
            .zip(self.slots.iter().copied())
            .filter(|(_, value)| !value.is_nil())
            .collect();
        let length_hint = self.length_hint.get();
        let hash_size = entries.len() + self.nodes.iter().filter(|node| !node.value.is_nil()).count();
        self.resize(self.array.len(), hash_size.max(1), headroom)?;
        self.shaped = false;
        self.shape = ROOT;
        self.slots = Vec::new();
        self.length_hint.set(length_hint);
        for (key, value) in entries {
            self.insert_new(Value::String(key.string), key.hash, value, None)?;
        }
        Ok(())
    }

    pub(crate) fn is_absent(&self, bit: u8) -> bool {
        self.absent & bit != 0
    }

    pub(crate) fn forget_absences(&mut self) {
        self.absent = 0;
    }

    pub(crate) fn mark_absent(&mut self, bit: u8) {
        self.absent |= bit;
    }

    fn forget_absence(&mut self, keys: KeyView, key: Value) {
        if self.absent == 0 {
            return;
        }
        if let Value::String(string) = key
            && keys.get(string).bytes.starts_with(b"__")
        {
            self.absent = 0;
        }
    }

    pub(crate) fn set_array_items(&mut self, first: usize, values: &[Value], headroom: Option<usize>) -> Result<(), SetError> {
        let start = first - 1;
        let end = start + values.len();
        self.grow_array(end, headroom)?;
        self.array[start..end].copy_from_slice(values);
        Ok(())
    }

    pub fn ensure_array(&mut self, size: usize) -> Result<(), SetError> {
        self.grow_array(size, None)
    }

    fn grow_array(&mut self, size: usize, headroom: Option<usize>) -> Result<(), SetError> {
        if size <= self.array.len() {
            return Ok(());
        }
        if self.nodes.is_empty() {
            return self.extend_array(size, headroom);
        }
        let hash_size = self.nodes.len();
        self.resize(size, hash_size, headroom)
    }

    fn extend_array(&mut self, size: usize, headroom: Option<usize>) -> Result<(), SetError> {
        if let Some(room) = headroom {
            let growth = Table::sized_bytes(size, 0) as i128 - Table::sized_bytes(self.array.len(), 0) as i128;
            if growth > room as i128 {
                return Err(SetError::Memory);
            }
        }
        self.reserve_array(size)?;
        self.array.resize(size, Value::Nil);
        self.last_free = 0;
        self.length_hint.set(size / 2);
        Ok(())
    }

    #[inline]
    pub(crate) fn reserve_slot(&mut self, headroom: Option<usize>) -> Result<usize, SetError> {
        if self.slots.len() < self.slots.capacity() {
            return Ok(0);
        }
        let additional = self.slots.capacity().max(4);
        let growth = additional * std::mem::size_of::<Value>();
        if headroom.is_some_and(|room| growth > room) {
            return Err(SetError::Memory);
        }
        self.slots.try_reserve_exact(additional).map_err(|_| SetError::Memory)?;
        Ok(growth)
    }

    /// Follows `luaH_getn` in ltable.c; the result becomes the next length hint.
    pub fn border(&self) -> i64 {
        let size = self.array.len();
        if size > 0
            && let Some(border) = self.array_border(size)
        {
            return border;
        }
        if self.nodes.is_empty() || self.get_int(size as i64 + 1).is_nil() {
            return size as i64;
        }
        self.hash_border(size as i64)
    }

    fn array_border(&self, size: usize) -> Option<i64> {
        let mut limit = self.length_hint.get().max(1);
        if self.array_is_empty(limit) {
            for _ in 0..MAX_VICINITY {
                if limit <= 1 {
                    break;
                }
                limit -= 1;
                if !self.array_is_empty(limit) {
                    return Some(self.set_length_hint(limit));
                }
            }
            return Some(self.set_length_hint(self.array_search(0, limit)));
        }
        for _ in 0..MAX_VICINITY {
            if limit >= size {
                break;
            }
            limit += 1;
            if self.array_is_empty(limit) {
                return Some(self.set_length_hint(limit - 1));
            }
        }
        if self.array_is_empty(size) {
            return Some(self.set_length_hint(self.array_search(limit, size)));
        }
        self.length_hint.set(size);
        None
    }

    fn array_is_empty(&self, index: usize) -> bool {
        self.array[index - 1].is_nil()
    }

    fn array_search(&self, mut low: usize, mut high: usize) -> usize {
        while high - low > 1 {
            let middle = (low + high) / 2;
            if self.array_is_empty(middle) {
                high = middle;
            } else {
                low = middle;
            }
        }
        low
    }

    fn set_length_hint(&self, hint: usize) -> i64 {
        self.length_hint.set(hint);
        hint as i64
    }

    fn hash_border(&self, asize: i64) -> i64 {
        const MAX: u64 = i64::MAX as u64;
        let asize = asize as u64;
        let width = if asize == 0 { 0 } else { ceil_log2(asize as usize) as u32 };
        let mut rnd = HASH_SEARCH_SEED;
        let mask = (1u64 << width) - 1;
        let mut i = asize + 1;
        let increment = (rnd & mask) + 1;
        let mut j = if increment <= MAX - i { i + increment } else { i + 1 };
        rnd >>= width;
        while !self.get_int(j as i64).is_nil() {
            i = j;
            if j < MAX / 2 {
                j = j * 2 + (rnd & 1);
                rnd >>= 1;
            } else {
                j = MAX;
                if self.get_int(j as i64).is_nil() {
                    break;
                }
                return j as i64;
            }
        }
        while j - i > 1 {
            let middle = (i + j) / 2;
            if self.get_int(middle as i64).is_nil() {
                j = middle;
            } else {
                i = middle;
            }
        }
        i as i64
    }

    pub fn next(&self, keys: KeyView, key: Value) -> Result<Option<(Value, Value)>, ()> {
        let position = self.position_after(keys, key)?;
        Ok(self.entry_from(keys, position))
    }

    fn position_after(&self, keys: KeyView, key: Value) -> Result<usize, ()> {
        let array_len = self.array.len();
        match number::normalize_key(key) {
            Value::Nil => Ok(0),
            Value::Integer(index) if array_slot(index, array_len).is_some() => Ok(index as usize),
            key => {
                if let Some(slot) = self.string_slot(keys, key) {
                    return Ok(array_len + slot + 1);
                }
                if self.nodes.is_empty() {
                    return Err(());
                }
                let hash = key_hash(keys, key);
                match self.find(keys, key, hash, true) {
                    Some(node) => Ok(array_len + self.slots.len() + node + 1),
                    None => Err(()),
                }
            }
        }
    }

    fn string_slot(&self, keys: KeyView, key: Value) -> Option<usize> {
        let Value::String(string) = key else {
            return None;
        };
        if !self.shaped || !keys.is_interned(string) {
            return None;
        }
        self.slot_of(keys, string, keys.hash(string))
    }

    fn entry_from(&self, keys: KeyView, mut position: usize) -> Option<(Value, Value)> {
        let array_len = self.array.len();
        while position < array_len {
            let value = self.array[position];
            if !value.is_nil() {
                return Some((Value::Integer(position as i64 + 1), value));
            }
            position += 1;
        }
        let node_base = array_len + self.slots.len();
        if position < node_base {
            let shape = keys.shapes.live(self.shape);
            for slot in position - array_len..self.slots.len() {
                let value = self.slots[slot];
                if !value.is_nil() {
                    return Some((Value::String(shape.key_at(keys.shapes, slot).string), value));
                }
            }
            position = node_base;
        }
        self.nodes[position - node_base..]
            .iter()
            .find(|node| !node.value.is_nil())
            .map(|node| (node.key, node.value))
    }

    fn find(&self, keys: KeyView, key: Value, hash: u64, dead_ok: bool) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut index = self.main_position(key, hash);
        loop {
            let node = &self.nodes[index];
            if node.hash == hash && node_matches(keys, node, key, dead_ok) {
                return Some(index);
            }
            match node.next {
                0 => return None,
                next => index = next as usize - 1,
            }
        }
    }

    fn main_position(&self, key: Value, hash: u64) -> usize {
        let size = self.nodes.len();
        match key {
            Value::Integer(_) => (hash % ((size - 1) | 1) as u64) as usize,
            _ => hash as usize & (size - 1),
        }
    }

    fn insert_new(&mut self, key: Value, hash: u64, value: Value, headroom: Option<usize>) -> Result<(), SetError> {
        if self.nodes.is_empty() {
            return self.rehash_and_insert(key, hash, value, headroom);
        }
        let main = self.main_position(key, hash);
        if self.nodes[main].value.is_nil() {
            self.nodes[main] = Node {
                key,
                value,
                hash,
                dead: false,
                ..self.nodes[main]
            };
            return Ok(());
        }
        let Some(free) = self.free_position() else {
            return self.rehash_and_insert(key, hash, value, headroom);
        };
        let colliding = self.nodes[main];
        let other = self.main_position(colliding.key, colliding.hash);
        if other == main {
            self.nodes[free] = Node {
                key,
                value,
                hash,
                next: colliding.next,
                dead: false,
            };
            self.nodes[main].next = free as u32 + 1;
            return Ok(());
        }
        let mut previous = other;
        while self.nodes[previous].next as usize != main + 1 {
            previous = self.nodes[previous].next as usize - 1;
        }
        self.nodes[previous].next = free as u32 + 1;
        self.nodes[free] = colliding;
        self.nodes[main] = Node {
            key,
            value,
            hash,
            next: 0,
            dead: false,
        };
        Ok(())
    }

    fn free_position(&mut self) -> Option<usize> {
        while self.last_free > 0 {
            self.last_free -= 1;
            if self.nodes[self.last_free].key.is_nil() {
                return Some(self.last_free);
            }
        }
        None
    }

    fn rehash_and_insert(&mut self, key: Value, hash: u64, value: Value, headroom: Option<usize>) -> Result<(), SetError> {
        self.rehash(key, headroom)?;
        if let Value::Integer(index) = key
            && let Some(slot) = array_slot(index, self.array.len())
        {
            self.array[slot] = value;
            return Ok(());
        }
        self.insert_new(key, hash, value, headroom)
    }

    fn rehash(&mut self, extra_key: Value, headroom: Option<usize>) -> Result<(), SetError> {
        let mut counts = Counters {
            total: 1,
            ..Counters::default()
        };
        counts.count_integer(extra_key);
        self.count_hash_keys(&mut counts);
        let array_size = if counts.array_keys == 0 {
            self.array.len()
        } else {
            self.count_array_keys(&mut counts);
            counts.compute_sizes()
        };
        let mut hash_size = counts.total - counts.array_keys;
        if counts.deleted {
            hash_size += hash_size >> 2;
        }
        self.resize(array_size, hash_size, headroom)
    }

    fn count_hash_keys(&self, counts: &mut Counters) {
        for node in &self.nodes {
            if node.value.is_nil() {
                counts.deleted = true;
                continue;
            }
            counts.total += 1;
            counts.count_integer(node.key);
        }
    }

    /// Follows `numusearray` in ltable.c.
    fn count_array_keys(&self, counts: &mut Counters) {
        let size = self.array.len();
        let mut used = 0;
        let mut start = 0;
        let mut limit = 1;
        for slice in &mut counts.slices {
            let end = limit.min(size);
            if start >= end {
                break;
            }
            let mut count = 0;
            for value in &self.array[start..end] {
                count += usize::from(!value.is_nil());
            }
            *slice += count;
            used += count;
            start = end;
            limit *= 2;
        }
        counts.total += used;
        counts.array_keys += used;
    }

    fn resize(&mut self, array_size: usize, hash_size: usize, headroom: Option<usize>) -> Result<(), SetError> {
        let hash_size = if self.shaped { hash_size } else { hash_size.max(1) };
        if let Some(room) = headroom {
            let growth = Table::sized_bytes(array_size, hash_size) as i128 - Table::sized_bytes(self.array.len(), self.nodes.len()) as i128;
            if growth > room as i128 {
                return Err(SetError::Memory);
            }
        }
        let nodes = nodes_for(hash_size)?;
        let old_size = self.array.len();
        if array_size > old_size {
            self.reserve_array(array_size)?;
        }
        let old_nodes = std::mem::replace(&mut self.nodes, nodes);
        self.last_free = self.nodes.len();
        if array_size < old_size {
            self.reinsert_old_slice(array_size)?;
        }
        self.array.resize(array_size, Value::Nil);
        self.length_hint.set(array_size / 2);
        for node in old_nodes.into_iter().filter(|node| !node.value.is_nil()) {
            match node.key {
                Value::Integer(index) => match array_slot(index, array_size) {
                    Some(slot) => self.array[slot] = node.value,
                    None => self.reinsert(node.key, node.hash, node.value)?,
                },
                _ => self.reinsert(node.key, node.hash, node.value)?,
            }
        }
        Ok(())
    }

    /// Follows `reinsertOldSlice` in ltable.c.
    fn reinsert_old_slice(&mut self, array_size: usize) -> Result<(), SetError> {
        for index in array_size..self.array.len() {
            let value = self.array[index];
            if value.is_nil() {
                continue;
            }
            self.reinsert(Value::Integer(index as i64 + 1), index as u64 + 1, value)?;
        }
        self.array.truncate(array_size);
        Ok(())
    }

    #[inline]
    fn reserve_array(&mut self, size: usize) -> Result<(), SetError> {
        if size <= self.array.capacity() {
            return Ok(());
        }
        self.reallocate_array(size)
    }

    /// Follows `resizearray` in ltable.c. A small array moves to a fresh block, which costs less than `realloc` moving it.
    #[inline(never)]
    fn reallocate_array(&mut self, size: usize) -> Result<(), SetError> {
        if size.saturating_mul(std::mem::size_of::<Value>()) > SMALL_ARRAY_BYTES {
            return self.array.try_reserve_exact(size - self.array.len()).map_err(|_| SetError::Memory);
        }
        let mut array = Vec::new();
        array.try_reserve_exact(size).map_err(|_| SetError::Memory)?;
        array.extend_from_slice(&self.array);
        self.array = array;
        Ok(())
    }

    fn reinsert(&mut self, key: Value, hash: u64, value: Value) -> Result<(), SetError> {
        self.insert_new(key, hash, value, None)
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn slot_value(&self, index: usize) -> Value {
        self.slots[index]
    }

    pub(crate) fn clear_slot_value(&mut self, index: usize) {
        self.slots[index] = Value::Nil;
    }

    pub(crate) fn set_slot(&mut self, index: usize, value: Value) {
        self.slots[index] = value;
    }
}

#[inline]
fn reserved_values(count: usize) -> Result<Vec<Value>, SetError> {
    let mut values = Vec::new();
    if count > 0 {
        values.try_reserve_exact(count).map_err(|_| SetError::Memory)?;
    }
    Ok(values)
}

#[inline]
fn filled_values(size: usize, spare: usize) -> Result<Vec<Value>, SetError> {
    let mut values = reserved_values(size + spare)?;
    values.resize(size, Value::Nil);
    Ok(values)
}

fn nodes_for(size: usize) -> Result<Vec<Node>, SetError> {
    if size == 0 {
        return Ok(Vec::new());
    }
    let size = size.next_power_of_two();
    let mut nodes = Vec::new();
    nodes.try_reserve_exact(size).map_err(|_| SetError::Memory)?;
    nodes.resize(size, Node::EMPTY);
    Ok(nodes)
}

impl Table {
    pub(crate) fn size(&self) -> usize {
        64 + std::mem::size_of::<Value>() * (self.array.capacity() + self.slots.capacity())
            + std::mem::size_of::<Node>() * self.nodes.capacity()
    }

    pub(crate) fn array_len(&self) -> usize {
        self.array.len()
    }

    pub(crate) fn array_value(&self, index: usize) -> Value {
        self.array[index]
    }

    pub(crate) fn array_values(&self) -> &[Value] {
        &self.array
    }

    pub(crate) fn slot_values(&self) -> &[Value] {
        &self.slots
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = (Value, Value)> + '_ {
        self.nodes.iter().map(|node| (node.key, node.value))
    }

    pub(crate) fn clear_array_value(&mut self, index: usize) {
        self.array[index] = Value::Nil;
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn node(&self, index: usize) -> (Value, Value) {
        let node = &self.nodes[index];
        (node.key, node.value)
    }

    pub(crate) fn clear_node_value(&mut self, index: usize) {
        self.nodes[index].value = Value::Nil;
    }

    pub(crate) fn node_is_dead(&self, index: usize) -> bool {
        self.nodes[index].dead
    }

    pub(crate) fn clear_node_key(&mut self, index: usize) {
        self.nodes[index].dead = true;
    }
}

fn node_matches(keys: KeyView, node: &Node, key: Value, dead_ok: bool) -> bool {
    if node.dead {
        return dead_ok && same_reference(node.key, key);
    }
    same_reference(node.key, key) || keys_equal(keys, node.key, key)
}

fn same_reference(stored: Value, key: Value) -> bool {
    match (stored, key) {
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Table(a), Value::Table(b)) => a == b,
        (Value::Closure(a), Value::Closure(b)) => a == b,
        (Value::Native(a), Value::Native(b)) => a == b,
        (Value::Thread(a), Value::Thread(b)) => a == b,
        (Value::Userdata(a), Value::Userdata(b)) => a == b,
        (Value::LightUserdata(a), Value::LightUserdata(b)) => a == b,
        _ => false,
    }
}

#[inline(always)]
fn array_slot(index: i64, len: usize) -> Option<usize> {
    let slot = (index as u64).wrapping_sub(1);
    (slot < len as u64).then_some(slot as usize)
}

#[derive(Default)]
struct Counters {
    total: usize,
    array_keys: usize,
    deleted: bool,
    slices: [usize; MAX_ARRAY_BITS + 1],
}

impl Counters {
    fn count_integer(&mut self, key: Value) {
        if let Value::Integer(index) = key
            && (1..=(1i64 << MAX_ARRAY_BITS)).contains(&index)
        {
            self.count_array_index(index as usize);
        }
    }

    fn count_array_index(&mut self, index: usize) {
        self.slices[ceil_log2(index)] += 1;
        self.array_keys += 1;
    }

    /// Follows `computesizes` in ltable.c.
    fn compute_sizes(&mut self) -> usize {
        let mut accumulated = 0;
        let mut in_array = 0;
        let mut optimal = 0;
        let mut power = 1usize;
        for &slice in &self.slices {
            if !array_pays_off(power, self.array_keys) {
                break;
            }
            accumulated += slice;
            if slice > 0 && array_pays_off(power, accumulated) {
                optimal = power;
                in_array = accumulated;
            }
            power *= 2;
        }
        self.array_keys = in_array;
        optimal
    }
}

fn array_pays_off(array_slots: usize, keys: usize) -> bool {
    array_slots <= keys * 3
}

fn ceil_log2(value: usize) -> usize {
    (usize::BITS - (value - 1).leading_zeros()) as usize
}

#[inline(always)]
fn key_hash(keys: KeyView, key: Value) -> u64 {
    match key {
        Value::Nil => 0,
        Value::Boolean(flag) => u64::from(flag),
        Value::Integer(index) => index as u64,
        Value::Float(f) => mix(f.to_bits()),
        Value::String(string) => keys.any_hash(string),
        Value::Table(reference) => mix(reference_index(reference)),
        Value::Closure(reference) => mix(reference_index(reference)),
        Value::Native(reference) => mix(reference_index(reference)),
        Value::Thread(reference) => mix(reference_index(reference)),
        Value::Userdata(reference) => mix(reference_index(reference)),
        Value::LightUserdata(pointer) => mix(pointer as u64),
    }
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^ (value >> 33)
}

fn reference_index<T>(reference: Ref<T>) -> u64 {
    reference.index() as u64
}

fn keys_equal(keys: KeyView, stored: Value, key: Value) -> bool {
    match (stored, key) {
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::String(a), Value::String(b)) => a == b || keys.same_content(a, b),
        (Value::Table(a), Value::Table(b)) => a == b,
        (Value::Closure(a), Value::Closure(b)) => a == b,
        (Value::Native(a), Value::Native(b)) => a == b,
        (Value::Thread(a), Value::Thread(b)) => a == b,
        (Value::Userdata(a), Value::Userdata(b)) => a == b,
        (Value::LightUserdata(a), Value::LightUserdata(b)) => a == b,
        _ => false,
    }
}
