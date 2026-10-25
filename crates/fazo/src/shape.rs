//! Follows V8's maps and LuaJIT's table templates.

use hashbrown::HashTable;

use crate::{
    arena::{Arena, Ref},
    heap::Heap,
    object::LuaString,
};

pub(crate) const MAX_KEYS: usize = 32;

pub(crate) const INDEX_MIN_KEYS: usize = 8;

/// Past this fan-out a table leaves shape mode, as V8 normalizes objects with too many map transitions.
pub(crate) const MAX_TRANSITIONS: usize = 64;

/// Dynamic-key stores build shapes that nothing else reuses, so each table gets a budget of them.
pub(crate) const MAX_DYNAMIC_TRANSITIONS: u8 = 8;

/// Arena slot 0; never released.
pub(crate) const ROOT: Ref<Shape> = Ref::root();

/// A dynamic key may not repeat, so its shapes are budgeted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeyOrigin {
    Constant,
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ShapeKey {
    pub(crate) string: Ref<LuaString>,
    pub(crate) hash: u64,
}

#[derive(Clone, Copy, Debug)]
struct SlotEntry {
    key: ShapeKey,
    slot: u32,
}

#[derive(Clone, Copy, Debug)]
struct Transition {
    key: ShapeKey,
    child: Ref<Shape>,
}

/// The key array and index are rebuilt on restore.
#[derive(Clone, Debug, Default)]
pub struct Shape {
    parent: Ref<Shape>,
    last: Option<ShapeKey>,
    len: u32,
    keys: Vec<ShapeKey>,
    index: Option<HashTable<SlotEntry>>,
    transitions: HashTable<Transition>,
}

#[derive(serde::Serialize)]
struct ShapeRecord {
    parent: Ref<Shape>,
    last: Option<ShapeKey>,
    len: u32,
}

#[derive(serde::Deserialize)]
struct OwnedShapeRecord {
    parent: Ref<Shape>,
    last: Option<ShapeKey>,
    len: u32,
}

impl serde::Serialize for Shape {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ShapeRecord {
            parent: self.parent,
            last: self.last,
            len: self.len,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let record = OwnedShapeRecord::deserialize(deserializer)?;
        Ok(Self {
            parent: record.parent,
            last: record.last,
            len: record.len,
            keys: Vec::new(),
            index: None,
            transitions: HashTable::new(),
        })
    }
}

impl Shape {
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    pub(crate) fn parent(&self) -> Ref<Shape> {
        self.parent
    }

    pub(crate) fn last_key(&self) -> Option<ShapeKey> {
        self.last
    }

    pub(crate) fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub(crate) fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<Transition>() * self.transitions.capacity()
            + std::mem::size_of::<ShapeKey>() * self.keys.capacity()
            + self.index.as_ref().map_or(0, |index| std::mem::size_of::<SlotEntry>() * index.capacity())
    }

    pub(crate) fn keys(&self, shapes: &Arena<Shape>) -> Vec<ShapeKey> {
        if !self.keys.is_empty() {
            return self.keys.clone();
        }
        let mut keys = Vec::with_capacity(self.len());
        let mut shape = self;
        while let Some(key) = shape.last {
            keys.push(key);
            shape = shapes.live(shape.parent);
        }
        keys.reverse();
        keys
    }

    pub(crate) fn key_at(&self, shapes: &Arena<Shape>, slot: usize) -> ShapeKey {
        if let Some(key) = self.keys.get(slot) {
            return *key;
        }
        let mut shape = self;
        while shape.len() > slot + 1 {
            shape = shapes.live(shape.parent);
        }
        shape.last.expect("a slot below the shape's length holds a key")
    }

    pub(crate) fn slot(&self, shapes: &Arena<Shape>, string: Ref<LuaString>, hash: u64) -> Option<usize> {
        if let Some(index) = &self.index {
            return index.find(hash, |entry| entry.key.string == string).map(|entry| entry.slot as usize);
        }
        let mut shape = self;
        while let Some(key) = shape.last {
            if key.string == string {
                return Some(shape.len() - 1);
            }
            shape = shapes.live(shape.parent);
        }
        None
    }

    /// Returns `None` when an ancestor is no longer alive.
    fn checked_keys(&self, shapes: &Arena<Shape>) -> Option<Vec<ShapeKey>> {
        let mut keys = Vec::with_capacity(self.len());
        let mut shape = self;
        while let Some(key) = shape.last {
            keys.push(key);
            shape = shapes.try_get(shape.parent)?;
        }
        keys.reverse();
        Some(keys)
    }

    fn install_keys(&mut self, keys: Vec<ShapeKey>) {
        let mut index = HashTable::with_capacity(keys.len());
        for (slot, key) in keys.iter().enumerate() {
            index.insert_unique(key.hash, SlotEntry { key: *key, slot: slot as u32 }, |entry| entry.key.hash);
        }
        self.keys = keys;
        self.index = Some(index);
    }

    fn child(&self, key: ShapeKey) -> Option<Ref<Shape>> {
        self.transitions
            .find(key.hash, |transition| transition.key == key)
            .map(|transition| transition.child)
    }

    fn extended(&self, parent: Ref<Shape>, key: ShapeKey, shapes: &Arena<Shape>) -> Self {
        let mut child = Self {
            parent,
            last: Some(key),
            len: self.len + 1,
            keys: Vec::new(),
            index: None,
            transitions: HashTable::new(),
        };
        if child.len() >= INDEX_MIN_KEYS {
            let mut keys = self.keys(shapes);
            keys.push(key);
            child.install_keys(keys);
        }
        child
    }

    pub(crate) fn add_transition(&mut self, key: ShapeKey, child: Ref<Shape>) {
        self.transitions
            .insert_unique(key.hash, Transition { key, child }, |transition| transition.key.hash);
    }

    pub(crate) fn forget_child(&mut self, child_key: ShapeKey, child: Ref<Shape>) {
        if let Ok(entry) = self
            .transitions
            .find_entry(child_key.hash, |transition| transition.child == child)
        {
            entry.remove();
        }
    }
}

impl Heap {
    pub(crate) fn existing_shape_child(&self, parent: Ref<Shape>, key: ShapeKey) -> Option<Ref<Shape>> {
        self.shapes.live(parent).child(key)
    }

    /// The transition is weak; the collector removes it when the child dies.
    pub(crate) fn create_shape_child(&mut self, parent: Ref<Shape>, key: ShapeKey) -> Ref<Shape> {
        let before = self.shapes.live(parent).size();
        let child = self.alloc(self.shapes.live(parent).extended(parent, key, &self.shapes));
        self.shapes.live_mut(parent).add_transition(key, child);
        let after = self.shapes.live(parent).size();
        self.grow_by(after, before);
        child
    }

    /// The root must be the heap's first object, at arena slot 0.
    pub(crate) fn init_shapes(&mut self) {
        let root = self.alloc(Shape::default());
        debug_assert_eq!(root, ROOT);
    }

    pub(crate) fn rebuild_transitions(&mut self) {
        let children: Vec<Ref<Shape>> = self.shapes.iter().map(|(reference, _)| reference).filter(|reference| *reference != ROOT).collect();
        for child in children {
            let shape = self.shapes.live(child);
            let (parent, key) = (shape.parent(), shape.last_key());
            if let (Some(key), true) = (key, self.shapes.try_get(parent).is_some()) {
                self.shapes.live_mut(parent).add_transition(key, child);
            }
            if self.shapes.live(child).len() < INDEX_MIN_KEYS {
                continue;
            }
            let Some(keys) = self.shapes.live(child).checked_keys(&self.shapes) else {
                continue;
            };
            let before = self.shapes.live(child).size();
            self.shapes.live_mut(child).install_keys(keys);
            let after = self.shapes.live(child).size();
            self.grow_by(after, before);
        }
    }
}
