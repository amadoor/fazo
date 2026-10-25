use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

/// Compiled code reads the index at offset 0 and compares the reference as one 64-bit word.
#[repr(C)]
pub struct Ref<T> {
    index: u32,
    generation: u32,
    kind: PhantomData<fn() -> T>,
}

impl<T> serde::Serialize for Ref<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.index, self.generation).serialize(serializer)
    }
}

impl<'de, T> serde::Deserialize<'de> for Ref<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (index, generation) = <(u32, u32)>::deserialize(deserializer)?;
        Ok(Self::new(index, generation))
    }
}

impl<T> Ref<T> {
    pub(crate) const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            kind: PhantomData,
        }
    }

    /// Tables with no keys use this shape; the arena keeps slot zero occupied by the first object inserted.
    pub(crate) const fn root() -> Self {
        Self::new(0, 0)
    }

    pub(crate) fn index(self) -> usize {
        self.index as usize
    }
}

impl<T> Default for Ref<T> {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ref<T> {}

impl<T> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool {
        self.packed() == other.packed()
    }
}

impl<T> Ref<T> {
    pub(crate) fn packed(self) -> u64 {
        u64::from(self.index) | (u64::from(self.generation) << 32)
    }
}

impl<T> Eq for Ref<T> {}

impl<T> Hash for Ref<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ref({}#{})", self.index, self.generation)
    }
}

const VACANT: u8 = 0x80;

/// Never issued, so a stale reference to a retired slot never matches.
const RETIRED: u32 = u32::MAX;

pub(crate) trait Slots<T>: Default {
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn push(&mut self, object: T);
    fn at(&self, index: usize) -> &T;
    fn at_mut(&mut self, index: usize) -> &mut T;
}

impl<T> Slots<T> for Vec<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    #[inline(always)]
    fn push(&mut self, object: T) {
        Vec::push(self, object);
    }

    #[inline(always)]
    fn at(&self, index: usize) -> &T {
        &self[index]
    }

    #[inline(always)]
    fn at_mut(&mut self, index: usize) -> &mut T {
        &mut self[index]
    }
}

/// Slots per chunk; a power of two.
const CHUNK: usize = 64;

/// Follows `luaF_newupval` in lfunc.c, which allocates each upvalue on its own; objects never move.
#[derive(Clone)]
pub(crate) struct Pinned<T> {
    chunks: Vec<Box<[T; CHUNK]>>,
    len: usize,
}

impl<T> Default for Pinned<T> {
    fn default() -> Self {
        Self { chunks: Vec::new(), len: 0 }
    }
}

#[inline(always)]
fn locate(index: usize) -> (usize, usize) {
    (index / CHUNK, index % CHUNK)
}

impl<T: Default> Slots<T> for Pinned<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn capacity(&self) -> usize {
        CHUNK * self.chunks.len()
    }

    #[inline]
    fn push(&mut self, object: T) {
        let (chunk, offset) = locate(self.len);
        if chunk == self.chunks.len() {
            self.chunks.push(Box::new(std::array::from_fn(|_| T::default())));
        }
        self.chunks[chunk][offset] = object;
        self.len += 1;
    }

    #[inline]
    fn at(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        let (chunk, offset) = locate(index);
        &self.chunks[chunk][offset]
    }

    #[inline]
    fn at_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        let (chunk, offset) = locate(index);
        &mut self.chunks[chunk][offset]
    }
}

impl<T: serde::Serialize + Default> serde::Serialize for Pinned<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((0..self.len).map(|index| self.at(index)))
    }
}

impl<'de, T: serde::Deserialize<'de> + Default> serde::Deserialize<'de> for Pinned<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let objects = Vec::<T>::deserialize(deserializer)?;
        let mut pinned = Self::default();
        for object in objects {
            pinned.push(object);
        }
        Ok(pinned)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Arena<T, S = Vec<T>> {
    objects: S,
    generations: Vec<u32>,
    #[serde(with = "crate::bytes")]
    marks: Vec<u8>,
    free: Vec<u32>,
    #[serde(skip)]
    kind: PhantomData<fn() -> T>,
}

impl<T, S: Slots<T>> Default for Arena<T, S> {
    fn default() -> Self {
        Self {
            objects: S::default(),
            generations: Vec::new(),
            marks: Vec::new(),
            free: Vec::new(),
            kind: PhantomData,
        }
    }
}

impl<T> Arena<T> {
    /// Invalid once the arena grows.
    pub(crate) fn objects_ptr(&self) -> *const T {
        self.objects.as_ptr()
    }

    pub(crate) fn objects_mut_ptr(&mut self) -> *mut T {
        self.objects.as_mut_ptr()
    }

    /// Borrowed apart so marks can change while objects are read.
    pub(crate) fn objects_and_marks(&mut self) -> (&[T], &mut [u8]) {
        (&self.objects, &mut self.marks)
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.objects
            .iter_mut()
            .zip(&self.marks)
            .filter(|(_, mark)| **mark != VACANT)
            .map(|(object, _)| object)
    }
}

impl<T: Default> Arena<T, Pinned<T>> {
    /// Stays valid until the object is freed.
    pub(crate) fn address(&mut self, reference: Ref<T>) -> *mut T {
        debug_assert!(self.holds(reference), "a reference the VM holds is live");
        let (chunk, offset) = locate(reference.index());
        self.objects.chunks[chunk].as_mut_ptr().wrapping_add(offset)
    }
}

impl<T, S: Slots<T>> Arena<T, S> {
    #[inline(always)]
    pub(crate) fn insert_marked(&mut self, object: T, marks: u8) -> Ref<T> {
        match self.free.pop() {
            Some(index) => {
                let slot = index as usize;
                *self.objects.at_mut(slot) = object;
                self.marks[slot] = marks;
                Ref::new(index, self.generations[slot])
            }
            None => self.push_marked(object, marks),
        }
    }

    #[inline(always)]
    fn push_marked(&mut self, object: T, marks: u8) -> Ref<T> {
        let index = u32::try_from(self.objects.len()).expect("arena index overflow");
        self.objects.push(object);
        self.generations.push(0);
        self.marks.push(marks);
        self.reserve_free_list();
        Ref::new(index, 0)
    }

    /// Keeps room in the free list for every slot, so a sweep never grows it.
    #[inline(always)]
    fn reserve_free_list(&mut self) {
        let slots = self.objects.capacity();
        if self.free.capacity() >= slots {
            return;
        }
        debug_assert!(self.free.is_empty(), "an arena grows only when no slot is free");
        self.free = Vec::with_capacity(slots);
    }

    pub fn get(&self, reference: Ref<T>) -> &T {
        self.try_get(reference).expect("stale heap reference")
    }

    pub fn get_mut(&mut self, reference: Ref<T>) -> &mut T {
        self.try_get_mut(reference).expect("stale heap reference")
    }

    pub fn try_get(&self, reference: Ref<T>) -> Option<&T> {
        self.holds(reference).then(|| self.objects.at(reference.index()))
    }

    pub fn try_get_mut(&mut self, reference: Ref<T>) -> Option<&mut T> {
        if !self.holds(reference) {
            return None;
        }
        Some(self.objects.at_mut(reference.index()))
    }

    pub(crate) fn live(&self, reference: Ref<T>) -> &T {
        debug_assert!(self.holds(reference), "a reference the VM holds is live");
        self.objects.at(reference.index())
    }

    pub(crate) fn live_mut(&mut self, reference: Ref<T>) -> &mut T {
        debug_assert!(self.holds(reference), "a reference the VM holds is live");
        self.objects.at_mut(reference.index())
    }

    fn holds(&self, reference: Ref<T>) -> bool {
        self.generations.get(reference.index()) == Some(&reference.generation)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Ref<T>, &T)> {
        self.generations
            .iter()
            .zip(&self.marks)
            .enumerate()
            .filter(|(_, (_, mark))| **mark != VACANT)
            .map(|(index, (generation, _))| (Ref::new(index as u32, *generation), self.objects.at(index)))
    }

    /// Invalid once the arena grows.
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn marks_ptr(&self) -> *const u8 {
        self.marks.as_ptr()
    }

    #[inline]
    pub(crate) fn marks(&self, reference: Ref<T>) -> Option<u8> {
        self.holds(reference).then(|| self.marks[reference.index()])
    }

    pub(crate) fn live_marks(&self, reference: Ref<T>) -> u8 {
        debug_assert!(self.holds(reference), "a reference the VM holds is live");
        self.marks[reference.index()]
    }

    pub(crate) fn marks_mut(&mut self, reference: Ref<T>) -> Option<&mut u8> {
        if !self.holds(reference) {
            return None;
        }
        Some(&mut self.marks[reference.index()])
    }

    #[inline]
    pub(crate) fn set_live_marks(&mut self, reference: Ref<T>, marks: u8) {
        debug_assert!(self.holds(reference), "a reference the VM holds is live");
        self.marks[reference.index()] = marks;
    }

    pub(crate) fn mark_bytes(&self) -> &[u8] {
        &self.marks
    }

    pub(crate) fn set_marks(&mut self, reference: Ref<T>, marks: u8) {
        if self.holds(reference) {
            self.marks[reference.index()] = marks;
        }
    }

    #[inline]
    pub(crate) fn sweep_slots(&mut self, slots: std::ops::Range<usize>, dead: u8, keep: u8, white: u8, mut free: impl FnMut(Ref<T>)) {
        for index in slots {
            let marks = self.marks[index];
            if marks == VACANT {
                continue;
            }
            if marks & dead != 0 {
                free(Ref::new(index as u32, self.generations[index]));
                continue;
            }
            self.marks[index] = (marks & keep) | white;
        }
    }

    #[inline]
    pub(crate) fn age_all(&mut self, dead: u8, age: impl Fn(u8) -> u8, mut free: impl FnMut(Ref<T>)) {
        for index in 0..self.marks.len() {
            let marks = self.marks[index];
            if marks == VACANT {
                continue;
            }
            if marks & dead != 0 {
                free(Ref::new(index as u32, self.generations[index]));
                continue;
            }
            self.marks[index] = age(marks);
        }
    }

    /// The caller guarantees slot `index` is occupied.
    pub(crate) fn object_at(&self, index: usize) -> &T {
        debug_assert!(self.marks[index] != VACANT);
        self.objects.at(index)
    }

    /// The caller guarantees slot `index` is occupied.
    pub(crate) fn marks_at_mut(&mut self, index: usize) -> &mut u8 {
        debug_assert!(self.marks[index] != VACANT);
        &mut self.marks[index]
    }

    /// The caller guarantees slot `index` is occupied.
    pub(crate) fn reference_at(&self, index: usize) -> Ref<T> {
        debug_assert!(self.marks[index] != VACANT);
        Ref::new(index as u32, self.generations[index])
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.objects.len()
    }

    pub(crate) fn occupied_at(&self, index: usize) -> Option<(Ref<T>, u8)> {
        let mark = *self.marks.get(index)?;
        (mark != VACANT).then(|| (Ref::new(index as u32, self.generations[index]), mark))
    }
}

impl<T: Default, S: Slots<T>> Arena<T, S> {
    /// The object is dropped in place, so it is never moved out of the arena.
    #[inline]
    pub(crate) fn release_with<R>(&mut self, reference: Ref<T>, measure: impl FnOnce(&T) -> R) -> Option<R> {
        if !self.holds(reference) {
            return None;
        }
        let index = reference.index();
        let measured = measure(self.objects.at(index));
        *self.objects.at_mut(index) = T::default();
        self.vacate(reference);
        Some(measured)
    }

    #[inline]
    fn vacate(&mut self, reference: Ref<T>) {
        let index = reference.index();
        self.marks[index] = VACANT;
        if self.generations[index] == RETIRED - 1 {
            self.generations[index] = RETIRED;
            return;
        }
        self.generations[index] += 1;
        self.free.push(reference.index);
    }
}

#[cfg(test)]
mod tests {
    use super::{Arena, Ref};

    #[test]
    fn a_slot_whose_next_generation_would_be_the_retired_one_is_never_reused() {
        let mut arena: Arena<u8> = Arena::default();
        arena.insert_marked(1, 0);
        arena.generations[0] = u32::MAX - 2;
        let stale = Ref::new(0, u32::MAX - 2);
        assert_eq!(arena.release_with(stale, |value| *value), Some(1));

        let last = arena.insert_marked(2, 0);
        assert_eq!(last, Ref::new(0, u32::MAX - 1));
        assert_eq!(arena.release_with(last, |value| *value), Some(2));

        assert!(arena.try_get(last).is_none());
        assert!(arena.try_get(stale).is_none());
        let fresh = arena.insert_marked(3, 0);
        assert_ne!(fresh.index(), 0);
        assert_eq!(arena.try_get(fresh), Some(&3));
        assert!(arena.try_get(last).is_none());
    }
}
