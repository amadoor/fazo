use super::naming;
use crate::{
    Vm,
    arena::Ref,
    object::{Closure, Frame, Resume, Thread},
    table::Table,
    value::Value,
};

const LEVELS_FIRST: i64 = 10;
const LEVELS_LAST: i64 = 11;

enum Level {
    Lua {
        closure: Ref<Closure>,
        pc: usize,
        tail: bool,
        hooked: bool,
    },
    Native { owner: Value, finalizer: bool },
}

impl Vm {
    pub(crate) fn traceback_of(&mut self, thread: Ref<Thread>, message: Option<&[u8]>, level: i64) -> Vec<u8> {
        let levels = self.stack_levels(thread);
        let last = levels.len() as i64;
        let start = level.max(0) + 1;
        let mut text = Vec::new();
        if let Some(message) = message {
            text.extend_from_slice(message);
            text.push(b'\n');
        }
        text.extend_from_slice(b"stack traceback:");
        let mut limit = if last - start > LEVELS_FIRST + LEVELS_LAST { LEVELS_FIRST } else { -1 };
        let mut current = start;
        while current <= last {
            let index = current as usize - 1;
            current += 1;
            if limit == 0 {
                let skipped = last - current - LEVELS_LAST + 1;
                text.extend_from_slice(format!("\n\t...\t(skipping {skipped} levels)").as_bytes());
                current += skipped;
            } else {
                self.push_level_line(&mut text, &levels, index);
            }
            limit -= 1;
        }
        text
    }

    fn push_level_line(&mut self, text: &mut Vec<u8>, levels: &[Level], index: usize) {
        match levels[index] {
            Level::Lua {
                closure,
                pc,
                tail,
                ..
            } => {
                let proto = self.proto_of(closure);
                let line = proto.lines.get(pc.saturating_sub(1)).copied().unwrap_or(proto.line_defined);
                let source = proto.source.to_vec();
                let main = proto.line_defined == 0;
                let defined = proto.line_defined;
                text.extend_from_slice(b"\n\t");
                text.extend_from_slice(&source);
                if line > 0 {
                    text.extend_from_slice(format!(":{line}").as_bytes());
                }
                text.extend_from_slice(b": in ");
                match self.level_name(levels, index, Value::Closure(closure)) {
                    Some(name) => text.extend_from_slice(&name),
                    None if main => text.extend_from_slice(b"main chunk"),
                    None => {
                        text.extend_from_slice(b"function <");
                        text.extend_from_slice(&source);
                        text.extend_from_slice(format!(":{defined}>").as_bytes());
                    }
                }
                if tail {
                    text.extend_from_slice(b"\n\t(...tail calls...)");
                }
            }
            Level::Native { owner, .. } => {
                text.extend_from_slice(b"\n\t[C]: in ");
                match self.level_name(levels, index, owner) {
                    Some(name) => text.extend_from_slice(&name),
                    None => text.push(b'?'),
                }
            }
        }
    }

    fn level_name(&mut self, levels: &[Level], index: usize, function: Value) -> Option<Vec<u8>> {
        if let Some(text) = self.call_site_text(levels, index) {
            return Some(text);
        }
        let name = self.global_function_name(function)?;
        let mut text = b"function '".to_vec();
        text.extend_from_slice(&name);
        text.push(b'\'');
        Some(text)
    }

    fn call_site_text(&self, levels: &[Level], index: usize) -> Option<Vec<u8>> {
        if matches!(levels[index], Level::Lua { tail: true, .. }) {
            return None;
        }
        let (kind, name) = self.call_site_name(levels, index)?;
        let mut text = kind.as_bytes().to_vec();
        text.extend_from_slice(b" '");
        text.extend_from_slice(&name);
        text.push(b'\'');
        Some(text)
    }

    fn call_site_name(&self, levels: &[Level], index: usize) -> Option<(&'static str, Vec<u8>)> {
        if let Some(Level::Native { finalizer: true, .. }) = levels.get(index + 1) {
            return Some(("metamethod", b"__gc".to_vec()));
        }
        let Some(Level::Lua {
            closure,
            pc,
            hooked,
            ..
        }) = levels.get(index + 1)
        else {
            return None;
        };
        if *hooked {
            return Some(("hook", b"?".to_vec()));
        }
        let proto = self.proto_of(*closure);
        naming::calling_name(&self.heap, proto, pc.checked_sub(1)?)
    }

    pub(crate) fn global_function_name(&mut self, function: Value) -> Option<Vec<u8>> {
        let modules = self.loaded_modules();
        for (module_name, module) in modules {
            for (key, value) in self.table_entries(module) {
                let Value::String(key) = key else {
                    continue;
                };
                if !self.raw_equal(value, function) {
                    continue;
                }
                let key = self.heap_bytes(key).to_vec();
                if module_name == b"_G" {
                    return Some(key);
                }
                let mut name = module_name;
                name.push(b'.');
                name.extend_from_slice(&key);
                return Some(name);
            }
        }
        None
    }

    fn loaded_modules(&mut self) -> Vec<(Vec<u8>, Ref<Table>)> {
        let globals = self.global_table();
        let package_key = self.string_value(b"package");
        let loaded_key = self.string_value(b"loaded");
        let loaded = match self.raw_get(globals, package_key) {
            Value::Table(package) => match self.table_get(package, loaded_key) {
                Value::Table(loaded) => Some(loaded),
                _ => None,
            },
            _ => None,
        };
        let Some(loaded) = loaded else {
            return match globals {
                Value::Table(table) => vec![(b"_G".to_vec(), table)],
                _ => Vec::new(),
            };
        };
        self.table_entries(loaded)
            .into_iter()
            .filter_map(|(name, module)| match (name, module) {
                (Value::String(name), Value::Table(module)) => Some((self.heap_bytes(name).to_vec(), module)),
                _ => None,
            })
            .collect()
    }

    fn table_entries(&mut self, table: Ref<Table>) -> Vec<(Value, Value)> {
        let mut entries = Vec::new();
        let mut key = Value::Nil;
        while let Ok(Some((next_key, value))) = self.table_next(table, key) {
            entries.push((next_key, value));
            key = next_key;
        }
        entries
    }

    fn stack_levels(&self, thread: Ref<Thread>) -> Vec<Level> {
        let running = self
            .current_native
            .filter(|_| thread == self.current_thread)
            .map(|call| Level::Native {
                owner: Value::Native(call.native),
                finalizer: false,
            });
        running
            .into_iter()
            .chain(
                self.heap
                    .threads
                    .get(thread)
                    .frames
                    .iter()
                    .rev()
                    .filter(|frame| !matches!(frame, Frame::Coroutine(_)))
                    .map(|frame| match frame {
                        Frame::Lua(lua) => Level::Lua {
                            closure: lua.closure,
                            pc: lua.pc(),
                            tail: lua.tail(),
                            hooked: lua.hooked(),
                        },
                        Frame::Native(native) => Level::Native {
                            owner: native.owner,
                            finalizer: matches!(&native.resume, Resume::Protected(protection) if protection.finalizer),
                        },
                        Frame::Yielded(yielded) => Level::Native {
                            owner: yielded.owner,
                            finalizer: false,
                        },
                        Frame::Coroutine(_) => unreachable!("coroutine entries are filtered out"),
                    }),
            )
            .collect()
    }
}
