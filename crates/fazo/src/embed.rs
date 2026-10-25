use crate::{
    LuaError, Values, Vm,
    stdlib::{
        base::{self, Piece},
        debug,
    },
    value::Value,
};

impl Vm {
    /// A syntax error comes back as the error object, a string.
    pub fn load(&mut self, chunkname: &[u8], source: &[u8]) -> Result<Value, LuaError> {
        self.compile_chunk(chunkname, source, None)
            .map_err(|message| self.error_string(message))
    }

    /// The error is the raw error object, without a traceback.
    pub fn call(&mut self, function: Value, arguments: &[Value]) -> Result<Values, LuaError> {
        self.call_value(function, arguments)
    }

    /// Errors come back with a traceback; the call uses two C-call levels, as the standalone interpreter's script call does.
    pub fn call_with_traceback(&mut self, function: Value, arguments: &[Value]) -> Result<Values, LuaError> {
        let handler = self.new_native("msghandler", debug::message_handler);
        self.enter_c_call()?;
        let outcome = self.call_protected(function, arguments, Some(handler));
        self.leave_c_call();
        outcome
    }

    pub fn string(&mut self, bytes: &[u8]) -> Value {
        self.string_value(bytes)
    }

    pub fn create_table(&mut self) -> Value {
        Value::Table(self.new_table())
    }

    /// Returns nil when `table` is not a table.
    pub fn raw_get(&self, table: Value, key: Value) -> Value {
        match table {
            Value::Table(table) => self.table_get(table, key),
            _ => Value::Nil,
        }
    }

    pub fn set_raw(&mut self, table: Value, key: Value, value: Value) -> Result<(), LuaError> {
        let Value::Table(table) = table else {
            return Err(self.error_string("table expected"));
        };
        self.raw_set(table, key, value)
    }

    /// Returns 0 when `table` is not a table.
    pub fn raw_len(&self, table: Value) -> i64 {
        match table {
            Value::Table(table) => self.table_border(table),
            _ => 0,
        }
    }

    /// Skips the protection `setmetatable` gives `__metatable`; `None` removes the metatable and any other value is an error.
    pub fn set_metatable(&mut self, table: Value, metatable: Option<Value>) -> Result<(), LuaError> {
        let Value::Table(table) = table else {
            return Err(self.error_string("table expected"));
        };
        let metatable = match metatable {
            None | Some(Value::Nil) => None,
            Some(Value::Table(metatable)) => Some(metatable),
            Some(_) => return Err(self.error_string("nil or table expected")),
        };
        self.set_table_metatable_checked(table, metatable);
        Ok(())
    }

    pub fn tostring(&mut self, value: Value) -> Result<Vec<u8>, LuaError> {
        match base::tostring_piece(self, value) {
            Piece::Text(text) => Ok(text),
            Piece::Call(handler) => {
                let results = self.call(handler, &[value])?;
                let result = results.first().copied().unwrap_or(Value::Nil);
                base::metamethod_text(self, result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::{LuaError, Options, Outcome, Value, Values, Vm};

    fn assert_send<T: Send>() {}

    fn record_running_native(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
        let name = vm.running_native().unwrap_or_default().to_owned();
        if let Some(recorded) = vm
            .host_data_mut()
            .and_then(|data| data.downcast_mut::<String>())
        {
            *recorded = name;
        }
        Ok(Values::new())
    }

    fn table_with_one_key(vm: &mut Vm) -> Value {
        let table = vm.create_table();
        let key = vm.string(b"answer");
        let value = Value::Integer(42);
        vm.set_raw(table, key, value).expect("a string key is valid");
        table
    }

    #[test]
    fn the_vm_is_send() {
        assert_send::<Vm>();
    }

    #[test]
    fn registry_values_survive_a_full_collection() {
        let mut vm = Vm::new(Options::default());
        let table = table_with_one_key(&mut vm);
        vm.set_registry_value("kept", table);
        vm.run_source(b"=test", b"collectgarbage('collect')").expect("the collection runs");
        let kept = vm.registry_value("kept");
        let key = vm.string(b"answer");
        assert!(matches!(vm.raw_get(kept, key), Value::Integer(42)));
        assert!(vm.registry_value("absent").is_nil());
    }

    #[test]
    fn string_bytes_read_string_values_only() {
        let mut vm = Vm::new(Options::default());
        let text = vm.string(b"a\0b");
        assert_eq!(vm.string_bytes(text), Some(&b"a\0b"[..]));
        assert_eq!(vm.string_bytes(Value::Integer(1)), None);
    }

    #[test]
    fn next_entry_visits_every_key_once() {
        let mut vm = Vm::new(Options::default());
        let table = vm.create_table();
        for (index, name) in [&b"one"[..], &b"two"[..], &b"three"[..]].into_iter().enumerate() {
            let key = vm.string(name);
            vm.set_raw(table, key, Value::Integer(index as i64)).expect("a string key is valid");
        }
        let mut visited = 0;
        let mut key = Value::Nil;
        while let Some((next, _)) = vm.next_entry(table, key).expect("the traversal continues") {
            visited += 1;
            key = next;
        }
        assert_eq!(visited, 3);
    }

    #[test]
    fn running_native_names_the_native_being_called() {
        let mut vm = Vm::new(Options::default());
        vm.register_native("probe.record", record_running_native).expect("the name is free");
        vm.set_host_data(Box::new(String::new()));
        let value = vm.native("probe.record").expect("the native is registered");
        let called = vm.call(value, &[]).expect("the native runs");
        assert!(called.is_empty());
        let recorded = vm
            .take_host_data()
            .and_then(|data: Box<dyn Any + Send>| data.downcast::<String>().ok())
            .expect("the host state is a string");
        assert_eq!(*recorded, "probe.record");
        assert!(vm.running_native().is_none());
    }

    #[test]
    fn host_data_is_replaced_and_taken() {
        let mut vm = Vm::new(Options::default());
        assert!(vm.host_data().is_none());
        vm.set_host_data(Box::new(7_u32));
        assert_eq!(vm.host_data().and_then(|data| data.downcast_ref::<u32>()), Some(&7));
        vm.set_host_data(Box::new(8_u32));
        assert_eq!(vm.host_data_mut().and_then(|data| data.downcast_mut::<u32>()).copied(), Some(8));
        assert!(vm.take_host_data().is_some());
        assert!(vm.host_data().is_none());
    }

    #[test]
    fn cancel_abandons_a_paused_call_and_the_vm_runs_on() {
        let mut vm = Vm::new(Options::default());
        let function = vm
            .load(b"=spin", b"local n = 0 while true do n = n + 1 end")
            .expect("the chunk compiles");
        vm.request_pause();
        assert!(matches!(vm.start(function, &[]), Ok(Outcome::Paused)));
        let message = vm.string(b"stopped");
        let error = vm.cancel(message).expect("a paused call is cancelled");
        assert_eq!(vm.error_message(&error), b"stopped");
        assert!(vm.resume().is_err());
        assert!(vm.cancel(message).is_none());
        vm.run_source(b"=after", b"x = 1").expect("the VM runs after a cancel");
    }

    #[test]
    fn owned_native_names_register_like_static_ones() {
        let mut vm = Vm::new(Options::default());
        let name = format!("generated.{}", 1);
        let value = vm
            .register_native(name.clone(), record_running_native)
            .expect("the owned name is free");
        assert!(matches!(value, Value::Native(_)));
        assert!(vm.native(&name).is_some());
        assert!(vm.register_native(name, record_running_native).is_err());
    }
}
