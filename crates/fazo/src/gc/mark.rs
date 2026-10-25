use crate::{
    shape::Shape,
    arena::{Arena, Ref},
    heap::{Collectable, GrayList, Heap},
    object::{Frame, Thread, UpState, UpVal},
    table::Table,
    value::Value,
    vm::{MAX_STACK_SLOTS, MODE_ABSENCE_BIT, REGISTER_SPAN},
};

use crate::Values;
use super::{
    Gc, OLD, OLD1, Phase, TOUCHED1, TOUCHED2, age, is_black, is_old, is_white,
    set_black, set_gray, value_gc,
};

impl Heap {
    #[inline]
    pub(crate) fn mark_gc(&mut self, gc: Gc) {
        if let Gc::Table(table) = gc {
            return self.mark_table(table);
        }
        if self.marks_of(gc).is_some_and(is_white) {
            self.really_mark(gc);
        }
    }

    /// Follows `mark_gc` in lgc.c.
    fn mark_table(&mut self, table: Ref<Table>) {
        let Some(marks) = self.tables.marks(table) else {
            return;
        };
        if !is_white(marks) {
            return;
        }
        self.gc.marked += self.tables.get(table).size() as i64;
        self.tables.set_marks(table, set_gray(marks));
        self.gc.gray.push(Gc::Table(table));
    }

    pub(crate) fn mark_value(&mut self, value: Value) {
        if let Some(gc) = value_gc(value) {
            self.mark_gc(gc);
        }
    }

    /// Follows `reallymarkobject` in lgc.c.
    pub(crate) fn really_mark(&mut self, gc: Gc) {
        match gc {
            Gc::String(reference) => self.mark_black(reference),
            Gc::Table(reference) => self.mark_gray(reference),
            Gc::Closure(reference) => self.mark_gray(reference),
            Gc::Native(reference) => self.mark_gray(reference),
            Gc::Proto(reference) => self.mark_gray(reference),
            Gc::Thread(reference) => self.mark_gray(reference),
            Gc::Shape(reference) => self.mark_gray(reference),
            Gc::UpVal(upvalue) => {
                let Some(marks) = self.upvalues.marks(upvalue) else {
                    return;
                };
                self.gc.marked += Collectable::size(self.upvalues.live(upvalue)) as i64;
                self.mark_upvalue(upvalue, marks);
            }
            Gc::Userdata(userdata) => self.really_mark_userdata(userdata),
        }
    }

    #[inline(always)]
    fn mark_black<T: Collectable>(&mut self, reference: Ref<T>) {
        let arena = T::arena(self);
        let Some(marks) = arena.marks(reference) else {
            return;
        };
        let size = arena.live(reference).size() as i64;
        arena.set_live_marks(reference, set_black(marks));
        self.gc.marked += size;
    }

    #[inline(always)]
    fn mark_gray<T: Collectable>(&mut self, reference: Ref<T>) {
        let arena = T::arena(self);
        let Some(marks) = arena.marks(reference) else {
            return;
        };
        let size = arena.live(reference).size() as i64;
        arena.set_live_marks(reference, set_gray(marks));
        self.gc.marked += size;
        self.gc.gray.push(T::wrap(reference));
    }

    fn really_mark_userdata(&mut self, userdata: Ref<crate::object::Userdata>) {
        let Some(marks) = self.userdata.marks(userdata) else {
            return;
        };
        let contents = self.userdata.live(userdata);
        self.gc.marked += Collectable::size(contents) as i64;
        if !contents.user_values.is_empty() {
            self.link_gray(Gc::Userdata(userdata), GrayList::Gray);
            return;
        }
        if let Some(metatable) = contents.metatable {
            self.mark_gc(Gc::Table(metatable));
        }
        self.userdata.set_live_marks(userdata, set_black(marks));
    }

    fn mark_upvalue(&mut self, upvalue: Ref<UpVal>, marks: u8) {
        let value = match self.upvalues.get(upvalue).state {
            UpState::Open { thread, index } => {
                self.set_marks_of(Gc::UpVal(upvalue), set_gray(marks));
                self.threads
                    .try_get(thread)
                    .and_then(|thread| thread.stack.get(index).copied())
                    .unwrap_or(Value::Nil)
            }
            UpState::Closed(value) => {
                self.set_marks_of(Gc::UpVal(upvalue), set_black(marks));
                value
            }
        };
        self.mark_value(value);
    }

    #[inline]
    pub(crate) fn link_gray(&mut self, gc: Gc, list: GrayList) {
        let Some(marks) = self.marks_of(gc) else {
            return;
        };
        self.set_marks_of(gc, set_gray(marks));
        let target = match list {
            GrayList::Gray => &mut self.gc.gray,
            GrayList::Again => &mut self.gc.grayagain,
            GrayList::Weak => &mut self.gc.weak,
            GrayList::AllWeak => &mut self.gc.allweak,
            GrayList::Ephemeron => &mut self.gc.ephemeron,
        };
        target.push(gc);
    }

    /// Follows `propagateall` in lgc.c.
    pub(crate) fn propagate_all(&mut self) -> i64 {
        self.propagate_until(i64::MAX)
    }

    /// Follows the `propagatemark` loop of `singlestep` in lgc.c.
    pub(crate) fn propagate_until(&mut self, limit: i64) -> i64 {
        let mut work = 0;
        loop {
            let (stop, done) = self.scan().tables(limit - work);
            work += done;
            match stop {
                Stop::Done => return work,
                Stop::Pending { table, dead_keys } => self.finish_table(table, dead_keys),
                Stop::Other if self.learn_mode_absent() => {}
                Stop::Other => {
                    work += self.propagate_one();
                    if work >= limit {
                        return work;
                    }
                }
            }
        }
    }

    fn learn_mode_absent(&mut self) -> bool {
        let Some(&Gc::Table(table)) = self.gc.gray.last() else {
            return false;
        };
        let Some(metatable) = self.tables.live(table).metatable else {
            return false;
        };
        let name = self.events[crate::vm::Field::Mode.index()];
        self.metatable_field(metatable, name, MODE_ABSENCE_BIT);
        self.tables.live(metatable).is_absent(MODE_ABSENCE_BIT)
    }

    /// Follows `propagatemark` in lgc.c.
    pub(crate) fn propagate_one(&mut self) -> i64 {
        let Some(gc) = self.gc.gray.pop() else {
            return 0;
        };
        if let Gc::Table(table) = gc {
            let marks = self.tables.live_marks(table);
            self.tables.set_live_marks(table, set_black(marks));
            return self.traverse_table(table);
        }
        if let Some(marks) = self.marks_of(gc) {
            self.set_marks_of(gc, set_black(marks));
        }
        self.traverse(gc)
    }

    fn scan(&mut self) -> Scan<'_> {
        let Heap { tables, strings, shapes, closures, natives, threads, userdata, white, gc, .. } = self;
        let (tables, table_marks) = tables.objects_and_marks();
        let (strings, string_marks) = strings.objects_and_marks();
        let (shapes, shape_marks) = shapes.objects_and_marks();
        Scan {
            tables,
            table_marks,
            strings,
            string_marks,
            shapes,
            shape_marks,
            closures: closures.mark_bytes(),
            natives: natives.mark_bytes(),
            threads,
            userdata: userdata.mark_bytes(),
            gray: &mut gc.gray,
            grayagain: &mut gc.grayagain,
            marked: &mut gc.marked,
            white,
        }
    }

    fn finish_table(&mut self, table: Ref<Table>, dead_keys: bool) {
        if dead_keys {
            self.clear_dead_keys(table);
        }
        self.mark_left_white();
    }

    /// Follows `traversethread` in lgc.c.
    fn mark_stack(&mut self, thread: Ref<Thread>, live: usize) {
        let mut scan = self.scan();
        let threads = scan.threads;
        scan.values(&threads.live(thread).stack[..live]);
        self.mark_left_white();
    }

    /// Each object is checked again, since marking an earlier one can mark it.
    fn mark_left_white(&mut self) {
        let mut at = 0;
        while let Some(&gc) = self.white.get(at) {
            self.mark_white(gc);
            at += 1;
        }
        self.white.clear();
    }

    fn traverse(&mut self, gc: Gc) -> i64 {
        match gc {
            Gc::Table(table) => self.traverse_table(table),
            Gc::Userdata(userdata) => self.traverse_userdata(userdata),
            Gc::Closure(closure) => self.traverse_closure(closure),
            Gc::Native(native) => self.traverse_native(native),
            Gc::Proto(proto) => self.traverse_proto(proto),
            Gc::Thread(thread) => self.traverse_thread(thread),
            Gc::Shape(shape) => self.traverse_shape(shape),
            Gc::String(_) | Gc::UpVal(_) => 0,
        }
    }

    #[inline]
    fn gen_link(&mut self, gc: Gc) {
        let Some(marks) = self.marks_of(gc) else {
            return;
        };
        if age(marks) == TOUCHED1 {
            self.link_gray(gc, GrayList::Again);
        } else if age(marks) == TOUCHED2 {
            self.set_age(gc, OLD);
        }
    }

    /// Follows `gfasttm` in ltm.h; a set absence flag answers nil without a lookup.
    pub(crate) fn metatable_field(&mut self, metatable: Ref<Table>, name: Ref<crate::object::LuaString>, absence_bit: u8) -> Value {
        if self.tables.live(metatable).is_absent(absence_bit) {
            return Value::Nil;
        }
        let value = self.tables.live(metatable).get(self.key_view(), Value::String(name));
        if value.is_nil() {
            self.tables.live_mut(metatable).mark_absent(absence_bit);
        }
        value
    }

    fn weak_mode(&mut self, metatable: Option<Ref<Table>>) -> (bool, bool) {
        let Some(metatable) = metatable else {
            return (false, false);
        };
        let name = self.events[crate::vm::Field::Mode.index()];
        match self.metatable_field(metatable, name, crate::vm::MODE_ABSENCE_BIT) {
            Value::String(mode) => {
                let text = self.bytes(mode);
                (text.contains(&b'k'), text.contains(&b'v'))
            }
            _ => (false, false),
        }
    }

    fn traverse_table(&mut self, table: Ref<Table>) -> i64 {
        let contents = self.tables.live(table);
        let metatable = contents.metatable;
        let shape = contents.shape();
        let work = traversal_work(contents);

        if let Some(metatable) = metatable {
            self.mark_white(Gc::Table(metatable));
        }
        if let Some(shape) = shape {
            self.mark_shape(shape);
        }
        if metatable.is_none() {
            self.traverse_strong(table);
            return work;
        }
        let (weak_keys, weak_values) = self.weak_mode(metatable);
        match (weak_keys, weak_values) {
            (false, false) => self.traverse_strong(table),
            (false, true) => self.traverse_weak_values(table),
            (true, false) => {
                self.traverse_ephemeron(table, false);
            }
            (true, true) => {
                let list = if self.gc.phase == Phase::Propagate {
                    GrayList::Again
                } else {
                    GrayList::AllWeak
                };
                self.link_gray(Gc::Table(table), list);
            }
        }
        work
    }

    /// Follows `mark_gc` in lgc.c.
    #[inline]
    fn mark_shape(&mut self, shape: Ref<Shape>) {
        if is_white(self.shapes.live_marks(shape)) {
            self.really_mark(Gc::Shape(shape));
        }
    }

    /// A shape keeps its parent and keys alive; its child transitions are weak and pruned when a child is freed.
    fn traverse_shape(&mut self, shape: Ref<Shape>) -> i64 {
        let parent = self.shapes.get(shape).parent();
        self.mark_gc(Gc::Shape(parent));
        let last = self.shapes.get(shape).last_key();
        if let Some(key) = last {
            self.mark_gc(Gc::String(key.string));
        }
        self.gen_link(Gc::Shape(shape));
        2
    }

    fn traverse_strong(&mut self, table: Ref<Table>) {
        if let Some(Stop::Pending { table, dead_keys }) = self.scan().strong(table) {
            self.finish_table(table, dead_keys);
        }
    }

    /// Follows `mark_gc` in lgc.c.
    #[inline]
    fn mark_white(&mut self, gc: Gc) {
        match gc {
            Gc::Table(table) => {
                let marks = self.tables.live_marks(table);
                if !is_white(marks) {
                    return;
                }
                self.gc.marked += self.tables.live(table).size() as i64;
                self.tables.set_live_marks(table, set_gray(marks));
                self.gc.gray.push(gc);
            }
            Gc::String(string) => {
                let marks = self.strings.live_marks(string);
                if !is_white(marks) {
                    return;
                }
                self.gc.marked += Collectable::size(self.strings.live(string)) as i64;
                self.strings.set_live_marks(string, set_black(marks));
            }
            _ => self.mark_gc(gc),
        }
    }

    fn clear_dead_keys(&mut self, table: Ref<Table>) {
        let contents = self.tables.get_mut(table);
        for index in 0..contents.node_count() {
            if contents.node(index).1.is_nil() {
                contents.clear_node_key(index);
            }
        }
    }

    fn traverse_weak_values(&mut self, table: Ref<Table>) {
        let mut has_clears = self.tables.get(table).array_len() > 0;
        for index in 0..self.tables.get(table).slot_count() {
            let value = self.tables.get(table).slot_value(index);
            if !has_clears && self.is_cleared(value) {
                has_clears = true;
            }
        }
        let node_count = self.tables.get(table).node_count();
        for index in 0..node_count {
            let (key, value) = self.tables.get(table).node(index);
            if value.is_nil() {
                self.tables.get_mut(table).clear_node_key(index);
                continue;
            }
            self.mark_value(key);
            if !has_clears && self.is_cleared(value) {
                has_clears = true;
            }
        }
        let list = if self.gc.phase == Phase::Atomic && has_clears {
            GrayList::Weak
        } else {
            GrayList::Again
        };
        self.link_gray(Gc::Table(table), list);
    }

    pub(crate) fn traverse_ephemeron(&mut self, table: Ref<Table>, inverted: bool) -> bool {
        let mut marked = false;
        let mut has_clears = false;
        let mut has_white_white = false;
        let array_len = self.tables.get(table).array_len();
        for index in 0..array_len {
            let value = self.tables.get(table).array_value(index);
            if value_gc(value).is_some_and(|gc| self.marks_of(gc).is_some_and(is_white)) {
                marked = true;
                self.mark_value(value);
            }
        }
        for index in 0..self.tables.get(table).slot_count() {
            let value = self.tables.get(table).slot_value(index);
            if value_gc(value).is_some_and(|gc| self.marks_of(gc).is_some_and(is_white)) {
                marked = true;
                self.mark_value(value);
            }
        }
        let node_count = self.tables.get(table).node_count();
        for step in 0..node_count {
            let index = if inverted { node_count - 1 - step } else { step };
            let (key, value) = self.tables.get(table).node(index);
            if value.is_nil() {
                self.tables.get_mut(table).clear_node_key(index);
                continue;
            }
            if self.is_cleared(key) {
                has_clears = true;
                if value_gc(value).is_some_and(|gc| self.marks_of(gc).is_some_and(is_white)) {
                    has_white_white = true;
                }
            } else if value_gc(value).is_some_and(|gc| self.marks_of(gc).is_some_and(is_white)) {
                marked = true;
                self.mark_value(value);
            }
        }
        if self.gc.phase == Phase::Propagate {
            self.link_gray(Gc::Table(table), GrayList::Again);
        } else if has_white_white {
            self.link_gray(Gc::Table(table), GrayList::Ephemeron);
        } else if has_clears {
            self.link_gray(Gc::Table(table), GrayList::AllWeak);
        } else {
            self.gen_link(Gc::Table(table));
        }
        marked
    }

    pub(crate) fn is_cleared(&mut self, value: Value) -> bool {
        match value_gc(value) {
            None => false,
            Some(Gc::String(string)) => {
                self.mark_gc(Gc::String(string));
                false
            }
            Some(gc) => self.marks_of(gc).is_none_or(is_white),
        }
    }

    pub(crate) fn converge_ephemerons(&mut self) -> i64 {
        let mut work = 0;
        let mut inverted = false;
        loop {
            let pending = std::mem::take(&mut self.gc.ephemeron);
            let mut changed = false;
            for gc in pending.iter().rev() {
                let Gc::Table(table) = *gc else {
                    continue;
                };
                if let Some(marks) = self.marks_of(*gc) {
                    self.set_marks_of(*gc, set_black(marks));
                }
                if self.traverse_ephemeron(table, inverted) {
                    work += self.propagate_all();
                    changed = true;
                }
            }
            inverted = !inverted;
            if !changed {
                break;
            }
        }
        work
    }

    pub(crate) fn clear_by_keys(&mut self, tables: &[Gc]) {
        for gc in tables {
            let Gc::Table(table) = *gc else {
                continue;
            };
            let node_count = self.tables.get(table).node_count();
            for index in 0..node_count {
                if self.tables.get(table).node_is_dead(index) {
                    continue;
                }
                let (key, _) = self.tables.get(table).node(index);
                if self.is_cleared(key) {
                    self.tables.get_mut(table).clear_node_value(index);
                }
                if self.tables.get(table).node(index).1.is_nil() {
                    self.tables.get_mut(table).clear_node_key(index);
                }
            }
        }
    }

    pub(crate) fn clear_by_values(&mut self, tables: &[Gc]) {
        for gc in tables {
            let Gc::Table(table) = *gc else {
                continue;
            };
            let array_len = self.tables.get(table).array_len();
            for index in 0..array_len {
                let value = self.tables.get(table).array_value(index);
                if self.is_cleared(value) {
                    self.tables.get_mut(table).clear_array_value(index);
                }
            }
            for index in 0..self.tables.get(table).slot_count() {
                let value = self.tables.get(table).slot_value(index);
                if self.is_cleared(value) {
                    self.tables.get_mut(table).clear_slot_value(index);
                }
            }
            let node_count = self.tables.get(table).node_count();
            for index in 0..node_count {
                let (_, value) = self.tables.get(table).node(index);
                if self.is_cleared(value) {
                    self.tables.get_mut(table).clear_node_value(index);
                }
                if self.tables.get(table).node(index).1.is_nil() {
                    self.tables.get_mut(table).clear_node_key(index);
                }
            }
        }
    }

    fn traverse_userdata(&mut self, userdata: Ref<crate::object::Userdata>) -> i64 {
        let metatable = self.userdata.get(userdata).metatable;
        if let Some(metatable) = metatable {
            self.mark_gc(Gc::Table(metatable));
        }
        let count = self.userdata.get(userdata).user_values.len();
        for index in 0..count {
            let value = self.userdata.get(userdata).user_values[index];
            self.mark_value(value);
        }
        self.gen_link(Gc::Userdata(userdata));
        1 + count as i64
    }

    fn traverse_closure(&mut self, closure: Ref<crate::object::Closure>) -> i64 {
        let proto = self.closures.get(closure).proto;
        self.mark_gc(Gc::Proto(proto));
        let count = self.closures.get(closure).upvalues.len();
        for index in 0..count {
            let upvalue = self.closures.get(closure).upvalues[index].reference;
            self.mark_gc(Gc::UpVal(upvalue));
        }
        1 + count as i64
    }

    fn traverse_native(&mut self, native: Ref<crate::object::Native>) -> i64 {
        let count = self.natives.get(native).upvalues.len();
        for index in 0..count {
            let value = self.natives.get(native).upvalues[index];
            self.mark_value(value);
        }
        1 + count as i64
    }

    fn traverse_proto(&mut self, proto: Ref<crate::proto::Proto>) -> i64 {
        let constants = self.protos.get(proto).constants.len();
        for index in 0..constants {
            let value = self.protos.get(proto).constants[index];
            self.mark_value(value);
        }
        let children = self.protos.get(proto).protos.len();
        for index in 0..children {
            let child = self.protos.get(proto).protos[index];
            self.mark_gc(Gc::Proto(child));
        }
        let mut shapes = std::mem::take(&mut self.white);
        let contents = self.protos.get(proto);
        let upvalue_names = contents.upvalues.iter().filter_map(|upvalue| upvalue.name);
        let local_names = contents.local_vars.iter().filter_map(|var| var.name);
        shapes.extend(upvalue_names.chain(local_names).map(Gc::String));
        for cache in &self.protos.get(proto).exec.feedback {
            cache.push_shapes(&mut shapes);
        }
        #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
        shapes.extend(self.protos.get(proto).jit.shapes.iter().map(|&shape| Gc::Shape(shape)));
        for gc in shapes.drain(..) {
            self.mark_gc(gc);
        }
        self.white = shapes;
        let proto = self.protos.get(proto);
        (1 + constants + proto.upvalues.len() + children + proto.local_vars.len()) as i64
    }

    fn traverse_thread(&mut self, thread: Ref<Thread>) -> i64 {
        let marks = self.threads.marks(thread).unwrap_or(0);
        if is_old(marks) || self.gc.phase == Phase::Propagate {
            self.link_gray(Gc::Thread(thread), GrayList::Again);
        }
        let extent = self.thread_extent(thread);
        let frame_count = self.threads.get(thread).frames.len();
        for index in 0..frame_count {
            self.mark_frame(thread, index);
        }
        let stack_len = self.threads.get(thread).stack.len();
        self.mark_stack(thread, extent.min(stack_len));
        let open_count = self.threads.get(thread).open_upvalues.len();
        for index in 0..open_count {
            let upvalue = self.threads.get(thread).open_upvalues[index];
            self.mark_gc(Gc::UpVal(upvalue));
        }
        let (failure, hook) = {
            let state = self.threads.get(thread);
            (state.failure, state.hook.map(|hook| hook.function))
        };
        for value in [failure, hook].into_iter().flatten() {
            self.mark_value(value);
        }
        if self.gc.phase == Phase::Atomic {
            if let Some(dead) = self.threads.get_mut(thread).stack.get_mut(extent..) {
                dead.fill(Value::Nil);
            }
            self.shrink_thread_stack(thread, extent);
        }
        1 + extent.min(stack_len) as i64
    }

    /// Follows `luaD_shrinkstack` in ldo.c; the live part includes the `REGISTER_SPAN` slots above the top frame's base.
    pub(crate) fn shrink_thread_stack(&mut self, thread: Ref<Thread>, extent: usize) {
        debug_assert!(self.threads.get(thread).open_upvalues.last().is_none_or(|&upvalue| {
            matches!(self.upvalues.get(upvalue).state, UpState::Open { index, .. } if index < extent)
        }));
        let inuse = extent + REGISTER_SPAN;
        if inuse > MAX_STACK_SLOTS {
            return;
        }
        let size = self.threads.get(thread).stack.len();
        let reasonable = if inuse > MAX_STACK_SLOTS / 3 { MAX_STACK_SLOTS } else { inuse * 3 };
        if size <= reasonable {
            return;
        }
        let target = if inuse > MAX_STACK_SLOTS / 2 { MAX_STACK_SLOTS } else { inuse * 2 };
        let state = self.threads.get_mut(thread);
        state.stack.truncate(target);
        state.stack.shrink_to_fit();
        self.stack_resized(thread);
        let released = (size - target) * std::mem::size_of::<Value>();
        self.gc.grow(-(released as i64));
    }

    pub(crate) fn thread_extent(&self, thread: Ref<Thread>) -> usize {
        let state = self.threads.get(thread);
        let count = state.frames.len();
        let mut extent = 0;
        let mut trust_top = true;
        for (index, frame) in state.frames.iter().enumerate() {
            let own = match frame {
                Frame::Lua(lua) if index + 1 == count => {
                    let closure = self.closures.get(lua.closure);
                    let limit = lua.base + usize::from(self.protos.get(closure.proto).max_stack);
                    let window = self.gc.safe_window.or(self.gc.live_native);
                    match window {
                        Some((running, slot, top)) if running == thread && (lua.base..limit).contains(&slot) => {
                            trust_top = false;
                            top
                        }
                        _ => limit,
                    }
                }
                Frame::Lua(_) | Frame::Yielded(_) | Frame::Coroutine(_) => 0,
                Frame::Native(native) => native.scratch,
            };
            let callee = state.frames.get(index + 1).map_or(0, frame_slot);
            extent = extent.max(own).max(callee);
        }
        if trust_top {
            extent = extent.max(state.top);
        }
        extent
    }

    /// Follows `markold` in lgc.c.
    pub(crate) fn mark_old_list(&mut self, list: &[Gc]) {
        for &gc in list {
            let black = match gc {
                Gc::String(reference) => self.age_old1(reference),
                Gc::Table(reference) => self.age_old1(reference),
                Gc::Closure(reference) => self.age_old1(reference),
                Gc::Native(reference) => self.age_old1(reference),
                Gc::UpVal(reference) => self.age_old1(reference),
                Gc::Proto(reference) => self.age_old1(reference),
                Gc::Thread(reference) => self.age_old1(reference),
                Gc::Userdata(reference) => self.age_old1(reference),
                Gc::Shape(reference) => self.age_old1(reference),
            };
            if black {
                self.really_mark(gc);
            }
        }
    }

    #[inline(always)]
    fn age_old1<T: Collectable>(&mut self, reference: Ref<T>) -> bool {
        let Some(slot) = T::arena(self).marks_mut(reference) else {
            return false;
        };
        let marks = *slot;
        if age(marks) != OLD1 {
            return false;
        }
        *slot = crate::gc::with_age(marks, OLD);
        is_black(marks)
    }

    pub(crate) fn mark_being_finalized(&mut self) -> i64 {
        let pending: Vec<Gc> = self.gc.tobefnz.iter().copied().collect();
        let count = pending.len() as i64;
        for gc in pending {
            self.mark_gc(gc);
        }
        count
    }

    pub(crate) fn separate_to_be_finalized(&mut self, all: bool) {
        if self.gc.finobj.is_empty() {
            return;
        }
        let finobj = std::mem::take(&mut self.gc.finobj);
        let mut kept = Vec::with_capacity(finobj.len());
        for gc in finobj.iter().rev() {
            let white = self.marks_of(*gc).is_none_or(is_white);
            if white || all {
                self.gc.tobefnz.push_back(*gc);
            } else {
                kept.push(*gc);
            }
        }
        kept.reverse();
        self.gc.finobj = kept;
    }

    /// Follows `correctgraylists` in lgc.c.
    pub(crate) fn correct_gray_lists(&mut self) {
        let mut list = std::mem::take(&mut self.gc.grayagain);
        list.append(&mut self.gc.weak);
        list.append(&mut self.gc.allweak);
        list.append(&mut self.gc.ephemeron);
        list.retain(|&gc| self.correct_gray(gc));
        self.gc.grayagain = list;
    }

    /// Follows `correctgraylist` in lgc.c.
    fn correct_gray(&mut self, gc: Gc) -> bool {
        let Some(marks) = self.marks_of(gc) else {
            return false;
        };
        if is_white(marks) {
            return false;
        }
        if age(marks) == TOUCHED1 {
            let marks = set_black(marks);
            self.set_marks_of(gc, marks);
            self.set_age(gc, TOUCHED2);
            return true;
        }
        if matches!(gc, Gc::Thread(_)) {
            return true;
        }
        if age(marks) == TOUCHED2 {
            self.set_age(gc, OLD);
        }
        let marks = set_black(self.marks_of(gc).unwrap_or(marks));
        self.set_marks_of(gc, marks);
        false
    }

    fn remark_upvalues(&mut self) {
        for index in 0..self.threads.slot_count() {
            let Some((thread, marks)) = self.threads.occupied_at(index) else {
                continue;
            };
            if !is_white(marks) {
                continue;
            }
            let open = self.threads.get(thread).open_upvalues.clone();
            for upvalue in open {
                let alive = self.upvalues.marks(upvalue).is_some_and(|marks| !is_white(marks));
                let Some(UpState::Open { index: slot, .. }) = self.upvalues.try_get(upvalue).map(|cell| cell.state) else {
                    continue;
                };
                if alive {
                    let value = self.threads.get(thread).stack.get(slot).copied().unwrap_or(Value::Nil);
                    self.mark_value(value);
                }
            }
        }
    }

    pub(crate) fn atomic(&mut self, roots: &[Value]) -> i64 {
        let spare = std::mem::take(&mut self.gc.spare_gray);
        let mut saved_again = std::mem::replace(&mut self.gc.grayagain, spare);
        self.gc.phase = Phase::Atomic;
        for root in roots {
            self.mark_value(*root);
        }
        let mut work = self.propagate_all();
        self.remark_upvalues();
        work += self.propagate_all();
        self.gc.gray.append(&mut saved_again);
        self.gc.spare_gray = saved_again;
        work += self.propagate_all();
        work += self.converge_ephemerons();
        let weak = self.gc.weak.clone();
        let all_weak = self.gc.allweak.clone();
        self.clear_by_values(&weak);
        self.clear_by_values(&all_weak);
        let original_weak = self.gc.weak.len();
        let original_all = self.gc.allweak.len();
        self.separate_to_be_finalized(false);
        work += self.mark_being_finalized();
        work += self.propagate_all();
        work += self.converge_ephemerons();
        let ephemerons = self.gc.ephemeron.clone();
        let all_weak = self.gc.allweak.clone();
        self.clear_by_keys(&ephemerons);
        self.clear_by_keys(&all_weak);
        let weak_tail = self.gc.weak[original_weak..].to_vec();
        let all_tail = self.gc.allweak[original_all..].to_vec();
        self.clear_by_values(&weak_tail);
        self.clear_by_values(&all_tail);
        self.gc.current_white = self.gc.other_white();
        work
    }
}

/// Follows `traversetable` in lgc.c; a table without hash keys still counts one hash node.
fn traversal_work(table: &Table) -> i64 {
    let nodes = (table.slot_count() + table.node_count()).next_power_of_two();
    (1 + 2 * nodes + table.array_len()) as i64
}

struct Scan<'a> {
    tables: &'a [Table],
    table_marks: &'a mut [u8],
    strings: &'a [crate::object::LuaString],
    string_marks: &'a mut [u8],
    shapes: &'a [Shape],
    shape_marks: &'a mut [u8],
    closures: &'a [u8],
    natives: &'a [u8],
    threads: &'a Arena<Thread>,
    userdata: &'a [u8],
    gray: &'a mut Vec<Gc>,
    grayagain: &'a mut Vec<Gc>,
    marked: &'a mut i64,
    white: &'a mut Vec<Gc>,
}

enum Stop {
    Done,
    Other,
    Pending { table: Ref<Table>, dead_keys: bool },
}

impl Scan<'_> {
    /// Follows `traversetable` in lgc.c for strong tables.
    fn tables(&mut self, limit: i64) -> (Stop, i64) {
        let tables = self.tables;
        let mut work = 0;
        while work < limit {
            let Some(&top) = self.gray.last() else {
                return (Stop::Done, work);
            };
            let Gc::Table(table) = top else {
                return (Stop::Other, work);
            };
            let index = table.index();
            let contents = &tables[index];
            let metatable = contents.metatable;
            if metatable.is_some_and(|metatable| !tables[metatable.index()].is_absent(MODE_ABSENCE_BIT)) {
                return (Stop::Other, work);
            }
            self.gray.pop();
            self.table_marks[index] = set_black(self.table_marks[index]);
            work += traversal_work(contents);
            if let Some(metatable) = metatable {
                self.table(metatable);
            }
            if let Some(shape) = contents.shape() {
                self.shape(shape);
            }
            if let Some(stop) = self.strong(table) {
                return (stop, work);
            }
        }
        (Stop::Done, work)
    }

    /// Follows `traversestrongtable` in lgc.c.
    #[inline(always)]
    fn strong(&mut self, table: Ref<Table>) -> Option<Stop> {
        let tables = self.tables;
        let contents = &tables[table.index()];
        self.values(contents.array_values());
        self.values(contents.slot_values());
        let dead_keys = contents.node_count() > 0 && self.nodes(contents);
        self.gen_link(table);
        (dead_keys || !self.white.is_empty()).then_some(Stop::Pending { table, dead_keys })
    }

    fn gen_link(&mut self, table: Ref<Table>) {
        let index = table.index();
        let marks = self.table_marks[index];
        if age(marks) == TOUCHED1 {
            self.table_marks[index] = set_gray(marks);
            self.grayagain.push(Gc::Table(table));
        } else if age(marks) == TOUCHED2 {
            self.table_marks[index] = crate::gc::with_age(marks, OLD);
        }
    }

    fn shape(&mut self, shape: Ref<Shape>) {
        let index = shape.index();
        let marks = self.shape_marks[index];
        if !is_white(marks) {
            return;
        }
        *self.marked += self.shapes[index].size() as i64;
        self.shape_marks[index] = set_gray(marks);
        self.gray.push(Gc::Shape(shape));
    }

    #[inline(always)]
    fn values(&mut self, values: &[Value]) {
        let mut rest = values;
        loop {
            rest = skip_marked(rest, self.table_marks);
            let [value, tail @ ..] = rest else {
                return;
            };
            rest = tail;
            match *value {
                Value::Table(reference) => self.table(reference),
                value => self.value(value),
            }
        }
    }

    #[inline(never)]
    fn nodes(&mut self, contents: &Table) -> bool {
        let mut dead_keys = false;
        for (key, value) in contents.nodes() {
            if value.is_nil() {
                dead_keys = true;
                continue;
            }
            self.value(key);
            self.value(value);
        }
        dead_keys
    }

    #[inline(always)]
    fn table(&mut self, reference: Ref<Table>) {
        let index = reference.index();
        let marks = self.table_marks[index];
        if !is_white(marks) {
            return;
        }
        if !self.white.is_empty() {
            self.white.push(Gc::Table(reference));
            return;
        }
        *self.marked += self.tables[index].size() as i64;
        self.table_marks[index] = set_gray(marks);
        self.gray.push(Gc::Table(reference));
    }

    #[inline(always)]
    fn string(&mut self, reference: Ref<crate::object::LuaString>) {
        let index = reference.index();
        let marks = self.string_marks[index];
        if !is_white(marks) {
            return;
        }
        *self.marked += Collectable::size(&self.strings[index]) as i64;
        self.string_marks[index] = set_black(marks);
    }

    #[inline(never)]
    fn value(&mut self, value: Value) {
        let (gc, marks) = match value {
            Value::String(reference) => return self.string(reference),
            Value::Table(reference) => return self.table(reference),
            Value::Closure(reference) => (Gc::Closure(reference), self.closures[reference.index()]),
            Value::Native(reference) => (Gc::Native(reference), self.natives[reference.index()]),
            Value::Thread(reference) => (Gc::Thread(reference), self.threads.mark_bytes()[reference.index()]),
            Value::Userdata(reference) => (Gc::Userdata(reference), self.userdata[reference.index()]),
            Value::Nil | Value::Boolean(_) | Value::Integer(_) | Value::Float(_) | Value::LightUserdata(_) => return,
        };
        if is_white(marks) {
            self.white.push(gc);
        }
    }
}

#[inline(always)]
fn skip_marked<'v>(rest: &'v [Value], table_marks: &[u8]) -> &'v [Value] {
    let mut values = rest.iter();
    loop {
        let before = values.as_slice();
        let Some(value) = values.next() else {
            return before;
        };
        let marking = match *value {
            Value::Table(reference) => is_white(table_marks[reference.index()]),
            value => value.is_collectable(),
        };
        if marking {
            return before;
        }
    }
}

fn frame_slot(frame: &Frame) -> usize {
    match frame {
        Frame::Lua(lua) => lua.ret(),
        Frame::Native(native) => native.ret,
        Frame::Yielded(yielded) => yielded.ret,
        Frame::Coroutine(_) => 0,
    }
}

impl Heap {
    fn mark_frame(&mut self, thread: Ref<Thread>, index: usize) {
        let mut values = Values::new();
        match &self.threads.live(thread).frames[index] {
            Frame::Lua(lua) => values.push(Value::Closure(lua.closure)),
            Frame::Native(native) => {
                values.push(native.owner);
                native.resume.values(&mut values);
            }
            Frame::Yielded(yielded) => values.push(yielded.owner),
            Frame::Coroutine(entry) => values.extend([Value::Thread(entry.caller), entry.owner]),
        }
        for value in values {
            self.mark_value(value);
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    #[test]
    fn a_stack_grown_by_deep_recursion_shrinks_once_the_recursion_returns() {
        let mut vm = Vm::new(Options::default());
        vm.run_source(b"=t", b"local function rec(d) if d == 0 then return 0 end return 1 + rec(d - 1) end rec(150000)")
            .expect("the recursion runs");
        let grown = vm.heap.threads.get(vm.current_thread).stack.len();
        vm.run_source(b"=t", b"collectgarbage('collect') collectgarbage('collect')").expect("the collector runs");
        let shrunk = vm.heap.threads.get(vm.current_thread).stack.len();
        assert!(shrunk < grown / 2, "the stack shrank from {grown} to {shrunk} slots");
    }
}
