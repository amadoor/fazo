use crate::{
    LuaError, Vm,
    arena::Ref,
    compiler,
    object::{Frame, Thread},
    table::{SetError, Table},
    value::Value,
};

impl Vm {
    /// Slots below the top belong to the host and count toward stack overflow, as in `lua_settop`.
    pub fn set_stack_top(&mut self, top: usize) {
        self.ensure_stack(top);
        self.thread_mut().top = top;
    }

    pub(crate) fn global_table(&self) -> Value {
        Value::Table(self.globals)
    }

    pub(crate) fn new_table(&mut self) -> Ref<Table> {
        self.heap.alloc(Table::default())
    }

    pub(crate) fn new_table_with_sizes(&mut self, array: usize, hash: usize) -> Result<Ref<Table>, LuaError> {
        match Table::with_sizes(array, hash) {
            Ok(table) => Ok(self.heap.alloc(table)),
            Err(_) => Err(self.memory_error()),
        }
    }

    pub(crate) fn registry(&self) -> Ref<Table> {
        self.registry
    }

    pub(crate) fn set_registry_slots(&mut self, main_thread: Ref<Thread>, globals: Ref<Table>) {
        let registry = self.registry;
        for (index, value) in [(1, Value::Thread(main_thread)), (2, Value::Table(globals))] {
            self.heap
                .table_set(registry, Value::Integer(index), value)
                .expect("integer keys are never nil or NaN");
        }
    }

    pub(crate) fn set_field(&mut self, table: Ref<Table>, name: &str, value: Value) {
        let key = self.heap.string_value(name.as_bytes());
        self.heap.table_set_field(table, key, value).expect("string keys are never nil or NaN");
    }

    pub(crate) fn set_hidden(&mut self, name: &'static str, value: Value) {
        self.hidden.push((name.to_owned(), value));
    }

    pub(crate) fn hidden(&self, name: &str) -> Value {
        self.hidden
            .iter()
            .find(|(key, _)| key == name)
            .map_or(Value::Nil, |(_, value)| *value)
    }

    pub(crate) fn heap_bytes(&self, string: Ref<crate::object::LuaString>) -> &[u8] {
        self.heap.bytes(string)
    }

    pub(crate) fn string_length(&self, string: Ref<crate::object::LuaString>) -> usize {
        self.heap.bytes(string).len()
    }

    pub(crate) fn raw_equal(&self, left: Value, right: Value) -> bool {
        self.heap.raw_equal(left, right)
    }

    pub(crate) fn table_border(&self, table: Ref<Table>) -> i64 {
        self.heap.tables.get(table).border()
    }

    pub(crate) fn table_get(&self, table: Ref<Table>, key: Value) -> Value {
        self.heap.tables.get(table).get(self.heap.key_view(), key)
    }

    pub(crate) fn table_set(&mut self, table: Ref<Table>, key: Value, value: Value) -> Result<(), LuaError> {
        match self.store_in_table(table, key, value) {
            Ok(()) => Ok(()),
            Err(SetError::NilKey) => Err(self.runtime_error_text("table index is nil")),
            Err(SetError::NanKey) => Err(self.runtime_error_text("table index is NaN")),
            Err(SetError::Memory) => Err(self.memory_error()),
        }
    }

    /// Retries after a full collection when growing the table fails, as PUC's allocator does.
    pub(crate) fn store_in_table(&mut self, table: Ref<Table>, key: Value, value: Value) -> Result<(), SetError> {
        match self.heap.table_set(table, key, value) {
            Err(SetError::Memory) => {
                let mark = self.pin([Value::Table(table), key, value]);
                self.gc_collect();
                self.unpin(mark);
                self.heap.table_set(table, key, value)
            }
            outcome => outcome,
        }
    }

    pub(crate) fn table_next(&mut self, table: Ref<Table>, key: Value) -> Result<Option<(Value, Value)>, LuaError> {
        match self.heap.tables.get(table).next(self.heap.key_view(), key) {
            Ok(entry) => Ok(entry),
            Err(()) => Err(self.error_string("invalid key to 'next'")),
        }
    }

    pub(crate) fn runtime_error_text(&mut self, message: &str) -> LuaError {
        self.runtime_error(message.as_bytes())
    }

    pub(crate) fn position_at_level(&self, level: usize) -> Option<Vec<u8>> {
        let frame = match self
            .thread()
            .frames
            .iter()
            .rev()
            .filter(|frame| !matches!(frame, Frame::Yielded(_) | Frame::Coroutine(_)))
            .nth(level - 1)?
        {
            Frame::Lua(lua) => *lua,
            Frame::Native(_) | Frame::Yielded(_) | Frame::Coroutine(_) => return None,
        };
        let proto = self.proto_of(frame.closure);
        let line = *proto.lines.get(frame.pc().saturating_sub(1))?;
        let mut prefix = proto.source.to_vec();
        prefix.extend_from_slice(format!(":{line}: ").as_bytes());
        Some(prefix)
    }

    pub(crate) fn compile_chunk(
        &mut self,
        chunkname: &[u8],
        source: &[u8],
        environment: Option<Value>,
    ) -> Result<Value, Vec<u8>> {
        self.compile_chunk_with(chunkname, source, environment, None)
    }

    pub(crate) fn compile_chunk_with(
        &mut self,
        chunkname: &[u8],
        source: &[u8],
        environment: Option<Value>,
        interrupt: Option<&crate::session::RequestHandle>,
    ) -> Result<Value, Vec<u8>> {
        let proto = compiler::compile(&mut self.heap, chunkname, source, interrupt)?;
        debug_assert_eq!(self.verify_proto(proto), Ok(()), "the compiler emits verified bytecode");
        let environment = environment.unwrap_or(Value::Table(self.globals));
        let upvalue = self.heap.alloc_closed_upvalue(environment);
        let captured = self.heap.captured(upvalue);
        let closure = self.heap.alloc_closure(proto, vec![captured]);
        Ok(Value::Closure(closure))
    }
}
