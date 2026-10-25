use crate::{
    arena::Ref,
    heap::{Collectable, Heap},
    object::{Thread, UpState, UpVal},
    shape::{ROOT, Shape},
    value::Value,
};

use super::{
    GC_BITS, Gc, NEW, OLD, OLD0, OLD1, SURVIVAL, WHITE_BITS, age, is_finalized, is_white, set_black, set_gray,
    with_age,
};

const ARENAS: usize = 9;

enum YoungStep {
    /// Freed or waiting for its finalizer, so it leaves the young list.
    Gone,
    Dead,
    Survives,
    Old1(i64),
}

impl Heap {
    pub(crate) fn free_gc(&mut self, gc: Gc) {
        match gc {
            Gc::String(reference) => {
                let interned = self.string_keys[reference.index()].interned;
                if let Some(bytes) = self.strings.release_with(reference, |string| string.bytes.len()) {
                    if interned {
                        self.forget_short_string(reference);
                    }
                    self.gc.grow(-(bytes as i64 + 40));
                }
            }
            Gc::Table(reference) => self.release_object(reference),
            Gc::Closure(reference) => self.release_object(reference),
            Gc::Native(reference) => self.release_object(reference),
            Gc::UpVal(reference) => self.free_upvalue(reference),
            Gc::Proto(reference) => self.release_object(reference),
            Gc::Thread(reference) => self.free_thread(reference),
            Gc::Userdata(reference) => self.release_object(reference),
            Gc::Shape(reference) => self.free_shape(reference),
        }
    }

    /// The root shape is never freed; a shape leaves its parent's transitions first, so no live shape names a freed child.
    fn free_shape(&mut self, shape: Ref<Shape>) {
        if shape == ROOT {
            return;
        }
        if let Some((parent, key)) = self.shapes.try_get(shape).map(|shape| (shape.parent(), shape.last_key()))
            && let (Some(key), Some(parent)) = (key, self.shapes.try_get_mut(parent))
        {
            parent.forget_child(key, shape);
        }
        self.release_object(shape);
    }

    fn release_object<T: Collectable>(&mut self, reference: Ref<T>) {
        if let Some(size) = T::arena(self).release_with(reference, Collectable::size) {
            self.gc.grow(-(size as i64));
        }
    }

    fn free_upvalue(&mut self, upvalue: Ref<UpVal>) {
        if let Some(UpState::Open { thread, .. }) = self.upvalues.try_get(upvalue).map(|cell| cell.state)
            && let Some(owner) = self.threads.try_get_mut(thread)
        {
            owner.open_upvalues.retain(|open| *open != upvalue);
        }
        self.release_object(upvalue);
    }

    fn free_thread(&mut self, thread: Ref<Thread>) {
        let open = self.threads.get(thread).open_upvalues.clone();
        let other_white = self.gc.other_white();
        for upvalue in open {
            let alive = self.upvalues.marks(upvalue).is_some_and(|marks| marks & other_white == 0);
            let Some(UpState::Open { index, .. }) = self.upvalues.try_get(upvalue).map(|cell| cell.state) else {
                continue;
            };
            if alive {
                let value = self.threads.get(thread).stack.get(index).copied().unwrap_or(Value::Nil);
                self.upvalues.get_mut(upvalue).close(value);
                if let Some(marks) = self.upvalues.marks(upvalue) {
                    self.set_marks_of(Gc::UpVal(upvalue), set_black(marks));
                }
            }
        }
        self.release_object(thread);
    }

    fn arena_slots(&self, arena: usize) -> usize {
        match arena {
            0 => self.strings.slot_count(),
            1 => self.tables.slot_count(),
            2 => self.closures.slot_count(),
            3 => self.natives.slot_count(),
            4 => self.upvalues.slot_count(),
            5 => self.protos.slot_count(),
            6 => self.threads.slot_count(),
            7 => self.userdata.slot_count(),
            _ => self.shapes.slot_count(),
        }
    }

    fn arena_occupied(&self, arena: usize, index: usize) -> Option<(Gc, u8)> {
        match arena {
            0 => self.strings.occupied_at(index).map(|(r, m)| (Gc::String(r), m)),
            1 => self.tables.occupied_at(index).map(|(r, m)| (Gc::Table(r), m)),
            2 => self.closures.occupied_at(index).map(|(r, m)| (Gc::Closure(r), m)),
            3 => self.natives.occupied_at(index).map(|(r, m)| (Gc::Native(r), m)),
            4 => self.upvalues.occupied_at(index).map(|(r, m)| (Gc::UpVal(r), m)),
            5 => self.protos.occupied_at(index).map(|(r, m)| (Gc::Proto(r), m)),
            6 => self.threads.occupied_at(index).map(|(r, m)| (Gc::Thread(r), m)),
            7 => self.userdata.occupied_at(index).map(|(r, m)| (Gc::Userdata(r), m)),
            _ => self.shapes.occupied_at(index).map(|(r, m)| (Gc::Shape(r), m)),
        }
    }

    pub(crate) fn arenas_swept(&self) -> bool {
        self.gc.sweep_arena >= ARENAS
    }

    /// Follows `sweepstep` in lgc.c; a run's dead objects are freed after its marks are read.
    pub(crate) fn sweep_arenas(&mut self, mut budget: usize) {
        while budget > 0 && self.gc.sweep_arena < ARENAS {
            let arena = self.gc.sweep_arena;
            let start = self.gc.sweep_index;
            let slots = self.arena_slots(arena);
            if start >= slots {
                self.gc.sweep_arena += 1;
                self.gc.sweep_index = 0;
                continue;
            }
            let count = budget.min(slots - start);
            self.sweep_run(arena, start..start + count);
            self.gc.sweep_index += count;
            budget -= count;
        }
    }

    fn sweep_run(&mut self, arena: usize, slots: std::ops::Range<usize>) {
        match arena {
            0 => self.sweep_slots::<crate::object::LuaString>(slots),
            1 => self.sweep_slots::<crate::table::Table>(slots),
            2 => self.sweep_slots::<crate::object::Closure>(slots),
            3 => self.sweep_slots::<crate::object::Native>(slots),
            4 => self.sweep_slots::<UpVal>(slots),
            5 => self.sweep_slots::<crate::proto::Proto>(slots),
            6 => self.sweep_slots::<Thread>(slots),
            7 => self.sweep_slots::<crate::object::Userdata>(slots),
            _ => self.sweep_slots::<Shape>(slots),
        }
    }

    fn sweep_slots<T: Collectable>(&mut self, slots: std::ops::Range<usize>) {
        let (dead, white) = (self.gc.other_white(), self.gc.current_white);
        let mut freed = std::mem::take(&mut self.white);
        T::arena(self).sweep_slots(slots, dead, !GC_BITS, white, |reference| freed.push(T::wrap(reference)));

        for gc in freed.drain(..) {
            self.free_gc(gc);
        }
        self.white = freed;
    }

    pub(crate) fn list_swept(&self, to_be_finalized: bool) -> bool {
        let len = if to_be_finalized {
            self.gc.tobefnz.len()
        } else {
            self.gc.finobj.len()
        };
        self.gc.sweep_index >= len
    }

    pub(crate) fn sweep_list(&mut self, to_be_finalized: bool, mut budget: usize) {
        let len = if to_be_finalized {
            self.gc.tobefnz.len()
        } else {
            self.gc.finobj.len()
        };
        while budget > 0 && self.gc.sweep_index < len {
            let gc = if to_be_finalized {
                self.gc.tobefnz[self.gc.sweep_index]
            } else {
                self.gc.finobj[self.gc.sweep_index]
            };
            self.gc.sweep_index += 1;
            if let Some(marks) = self.marks_of(gc) {
                let white = self.gc.current_white;
                self.set_marks_of(gc, (marks & !GC_BITS) | white);
            }
            budget -= 1;
        }
    }

    /// Follows `sweepgen` in lgc.c.
    pub(crate) fn sweep_young(&mut self, mut young: Vec<Gc>) -> i64 {
        let mut survivors = std::mem::take(&mut self.gc.young);
        survivors.clear();
        let mut added = 0;
        for gc in young.drain(..) {
            let step = match gc {
                Gc::String(reference) => self.sweep_young_object(reference),
                Gc::Table(reference) => self.sweep_young_object(reference),
                Gc::Closure(reference) => self.sweep_young_object(reference),
                Gc::Native(reference) => self.sweep_young_object(reference),
                Gc::UpVal(reference) => self.sweep_young_object(reference),
                Gc::Proto(reference) => self.sweep_young_object(reference),
                Gc::Thread(reference) => self.sweep_young_object(reference),
                Gc::Userdata(reference) => self.sweep_young_object(reference),
                Gc::Shape(reference) => self.sweep_young_object(reference),
            };
            match step {
                YoungStep::Gone => {}
                YoungStep::Dead => self.free_gc(gc),
                YoungStep::Survives => survivors.push(gc),
                YoungStep::Old1(size) => {
                    added += size;
                    self.gc.old1.push(gc);
                }
            }
        }
        self.gc.young = survivors;
        self.gc.spare_young = young;
        added
    }

    #[inline(always)]
    fn sweep_young_object<T: Collectable>(&mut self, reference: Ref<T>) -> YoungStep {
        let white = self.gc.current_white;
        let arena = T::arena(self);
        let Some(slot) = arena.marks_mut(reference) else {
            return YoungStep::Gone;
        };
        let marks = *slot;
        if is_finalized(marks) {
            return YoungStep::Gone;
        }
        if is_white(marks) {
            return YoungStep::Dead;
        }
        match age(marks) {
            NEW => {
                *slot = (marks & !GC_BITS) | SURVIVAL | white;
                YoungStep::Survives
            }
            SURVIVAL | OLD0 => {
                *slot = with_age(marks, OLD1);
                YoungStep::Old1(arena.live(reference).size() as i64)
            }
            _ => YoungStep::Survives,
        }
    }

    pub(crate) fn age_young_list(&mut self, list: Vec<Gc>) -> (Vec<Gc>, i64) {
        let white = self.gc.current_white;
        let mut added = 0;
        for gc in &list {
            let Some(marks) = self.marks_of(*gc) else {
                continue;
            };
            match age(marks) {
                NEW => self.set_marks_of(*gc, (marks & !GC_BITS) | SURVIVAL | white),
                SURVIVAL | OLD0 => {
                    added += self.object_size(*gc);
                    self.set_age(*gc, OLD1);
                    self.gc.old1.push(*gc);
                }
                _ => {}
            }
        }
        (list, added)
    }

    /// Follows `sweep2old` in lgc.c.
    pub(crate) fn sweep_all_to_old(&mut self) {
        self.sweep_arena_to_old::<crate::object::LuaString>();
        self.sweep_arena_to_old::<crate::table::Table>();
        self.sweep_arena_to_old::<crate::object::Closure>();
        self.sweep_arena_to_old::<crate::object::Native>();
        self.sweep_kept_gray_to_old(4);
        self.sweep_arena_to_old::<crate::proto::Proto>();
        self.sweep_kept_gray_to_old(6);
        self.sweep_arena_to_old::<crate::object::Userdata>();
        self.sweep_arena_to_old::<Shape>();
        for gc in self.gc.finobj.clone() {
            if let Some(marks) = self.marks_of(gc) {
                self.make_old(gc, marks);
            }
        }
        for gc in self.gc.tobefnz.clone() {
            if let Some(marks) = self.marks_of(gc) {
                self.make_old(gc, marks);
            }
        }
    }

    fn sweep_arena_to_old<T: Collectable>(&mut self) {
        let mut freed = std::mem::take(&mut self.white);
        T::arena(self).age_all(WHITE_BITS, |marks| set_black(with_age(marks, OLD)), |reference| freed.push(T::wrap(reference)));

        for gc in freed.drain(..) {
            self.free_gc(gc);
        }
        self.white = freed;
    }

    /// Open upvalues and threads stay gray when old.
    fn sweep_kept_gray_to_old(&mut self, arena: usize) {
        for index in 0..self.arena_slots(arena) {
            let Some((gc, marks)) = self.arena_occupied(arena, index) else {
                continue;
            };
            if is_white(marks) {
                self.free_gc(gc);
                continue;
            }
            self.make_old(gc, marks);
        }
    }

    fn make_old(&mut self, gc: Gc, marks: u8) {
        let marks = with_age(marks, OLD);
        match gc {
            Gc::Thread(_) => {
                self.set_marks_of(gc, set_gray(marks));
                self.gc.grayagain.push(gc);
            }
            Gc::UpVal(upvalue) if self.upvalues.try_get(upvalue).is_some_and(UpVal::is_open) => {
                self.set_marks_of(gc, set_gray(marks));
            }
            _ => self.set_marks_of(gc, set_black(marks)),
        }
    }
}
