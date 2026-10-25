use std::{
    borrow::Cow,
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

/// Output is written as produced, so a script that never returns still streams its output.
pub trait Stdio: Send {
    fn write_stdout(&mut self, bytes: &[u8]);
    fn write_stderr(&mut self, bytes: &[u8]);

    fn read_stdin(&mut self, buffer: &mut [u8]) -> usize;

    fn take_stdout(&mut self) -> Vec<u8> {
        Vec::new()
    }

    fn take_stderr(&mut self) -> Vec<u8> {
        Vec::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seek {
    Start(u64),
    Current(i64),
    End(i64),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostError {
    pub errno: i32,
    pub reason: String,
}

impl HostError {
    pub fn new(errno: i32, reason: &str) -> Self {
        Self {
            errno,
            reason: reason.to_owned(),
        }
    }

    pub fn not_found() -> Self {
        Self::new(ENOENT, "No such file or directory")
    }

    pub fn unsupported() -> Self {
        Self::new(ENOSYS, "Function not implemented")
    }
}

pub const ENOENT: i32 = 2;
pub const ENOSYS: i32 = 38;
const EBADF: i32 = 9;
const EINVAL: i32 = 22;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitStatus {
    Exited(i32),
    Signaled(i32),
}

pub trait HostFile: Send {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, HostError>;
    fn write(&mut self, bytes: &[u8]) -> Result<(), HostError>;
    fn seek(&mut self, position: Seek) -> Result<u64, HostError>;
    fn flush(&mut self) -> Result<(), HostError>;

    /// A process started by `popen` reports its exit status; a file reports none.
    fn finish(&mut self) -> Result<Option<ExitStatus>, HostError> {
        Ok(None)
    }
}

/// `mode` takes the C stdio forms, such as `r`, `wb` or `a+`.
pub trait FileSystem: Send {
    fn open(&mut self, path: &[u8], mode: &[u8]) -> Result<Box<dyn HostFile>, HostError>;
    fn remove(&mut self, path: &[u8]) -> Result<(), HostError>;
    fn rename(&mut self, from: &[u8], to: &[u8]) -> Result<(), HostError>;
    fn tmpname(&mut self) -> Result<Vec<u8>, HostError>;

    fn popen(&mut self, _command: &[u8], _mode: &[u8]) -> Result<Box<dyn HostFile>, HostError> {
        Err(HostError::unsupported())
    }
}

/// Fields follow C's `struct tm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeFields {
    pub year: i64,
    pub month: i64,
    pub day: i64,
    pub hour: i64,
    pub minute: i64,
    pub second: i64,
    pub weekday: i64,
    pub yearday: i64,
    pub daylight_saving: Option<bool>,
}

pub trait System: Send {
    fn now(&mut self) -> i64;

    fn processor_seconds(&mut self) -> f64;

    fn getenv(&mut self, name: &[u8]) -> Option<Vec<u8>>;

    fn break_down(&mut self, seconds: i64, utc: bool) -> Option<TimeFields>;

    fn exit(&mut self, code: i32);

    fn executable_directory(&mut self) -> Option<Vec<u8>>;

    fn shell_available(&mut self) -> bool {
        false
    }

    fn execute(&mut self, _command: &[u8]) -> Result<ExitStatus, HostError> {
        Err(HostError::unsupported())
    }
}

/// A host that restores a snapshot must provide every native the snapshot uses.
pub struct Host {
    pub stdio: Box<dyn Stdio>,
    pub files: Box<dyn FileSystem>,
    pub system: Box<dyn System>,
    natives: Vec<crate::object::NativeEntry>,
}

impl Host {
    pub fn new(stdio: Box<dyn Stdio>, files: Box<dyn FileSystem>, system: Box<dyn System>) -> Self {
        Self {
            stdio,
            files,
            system,
            natives: Vec::new(),
        }
    }

    /// Output is kept in memory and the clock is fixed, so runs repeat.
    pub fn captured() -> Self {
        Self::new(Box::new(Captured::default()), Box::new(NoFiles), Box::new(FixedSystem))
    }

    pub fn register_native(&mut self, name: impl Into<Cow<'static, str>>, function: crate::NativeFn) -> Result<(), RegistrationError> {
        self.register_native_with_data(name, function, 0)
    }

    pub fn register_native_with_data(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        function: crate::NativeFn,
        data: u32,
    ) -> Result<(), RegistrationError> {
        self.add_native(name.into(), crate::object::NativeBody::Plain(function), data)
    }

    pub fn register_stepped(&mut self, name: impl Into<Cow<'static, str>>, function: crate::SteppedFn) -> Result<(), RegistrationError> {
        self.add_native(name.into(), crate::object::NativeBody::Stepped(function), 0)
    }

    fn add_native(&mut self, name: Cow<'static, str>, body: crate::object::NativeBody, data: u32) -> Result<(), RegistrationError> {
        if self.natives.iter().any(|entry| entry.name == name) {
            return Err(RegistrationError { name: name.into_owned() });
        }
        self.natives.push(crate::object::NativeEntry { name, body, data, builtin: None });
        Ok(())
    }

    pub(crate) fn take_natives(&mut self) -> Vec<crate::object::NativeEntry> {
        std::mem::take(&mut self.natives)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationError {
    name: String,
}

impl RegistrationError {
    pub(crate) fn new(name: &str) -> Self {
        Self { name: name.to_owned() }
    }
}

impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the native function `{}` is already registered", self.name)
    }
}

impl std::error::Error for RegistrationError {}

/// Keeps output in memory; read it with `Stdio::take_stdout`, and set input with `with_stdin`.
#[derive(Default)]
pub struct Captured {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdin: Vec<u8>,
    position: usize,
}

impl Captured {
    pub fn with_stdin(stdin: Vec<u8>) -> Self {
        Self {
            stdin,
            ..Self::default()
        }
    }
}

impl Stdio for Captured {
    fn write_stdout(&mut self, bytes: &[u8]) {
        self.stdout.extend_from_slice(bytes);
    }

    fn write_stderr(&mut self, bytes: &[u8]) {
        self.stderr.extend_from_slice(bytes);
    }

    fn read_stdin(&mut self, buffer: &mut [u8]) -> usize {
        let available = &self.stdin[self.position..];
        let count = available.len().min(buffer.len());
        buffer[..count].copy_from_slice(&available[..count]);
        self.position += count;
        count
    }

    fn take_stdout(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.stdout)
    }

    fn take_stderr(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.stderr)
    }
}

/// Every open fails as a missing file.
pub struct NoFiles;

impl FileSystem for NoFiles {
    fn open(&mut self, _path: &[u8], _mode: &[u8]) -> Result<Box<dyn HostFile>, HostError> {
        Err(HostError::not_found())
    }

    fn remove(&mut self, _path: &[u8]) -> Result<(), HostError> {
        Err(HostError::not_found())
    }

    fn rename(&mut self, _from: &[u8], _to: &[u8]) -> Result<(), HostError> {
        Err(HostError::not_found())
    }

    fn tmpname(&mut self) -> Result<Vec<u8>, HostError> {
        Err(HostError::new(EINVAL, "unable to generate a unique filename"))
    }
}

/// Files are shared with every handle opened from them, so a test can inspect what a script wrote.
#[derive(Clone, Default)]
pub struct MemoryFiles {
    contents: Arc<Mutex<BTreeMap<Vec<u8>, Vec<u8>>>>,
    counter: Arc<Mutex<u64>>,
}

fn locked<T>(shared: &Mutex<T>) -> MutexGuard<'_, T> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MemoryFiles {
    pub fn with_file(self, path: &[u8], contents: &[u8]) -> Self {
        locked(&self.contents).insert(path.to_vec(), contents.to_vec());
        self
    }

    pub fn contents(&self, path: &[u8]) -> Option<Vec<u8>> {
        locked(&self.contents).get(path).cloned()
    }
}

struct MemoryFile {
    path: Vec<u8>,
    contents: Arc<Mutex<BTreeMap<Vec<u8>, Vec<u8>>>>,
    position: usize,
    writable: bool,
    readable: bool,
    append: bool,
}

impl FileSystem for MemoryFiles {
    fn open(&mut self, path: &[u8], mode: &[u8]) -> Result<Box<dyn HostFile>, HostError> {
        let kind = mode.first().copied().unwrap_or(b'r');
        let plus = mode.contains(&b'+');
        let exists = locked(&self.contents).contains_key(path);
        if kind == b'r' && !exists {
            return Err(HostError::not_found());
        }
        if kind == b'w' || (kind == b'a' && !exists) {
            locked(&self.contents).insert(path.to_vec(), Vec::new());
        }
        let position = if kind == b'a' {
            locked(&self.contents).get(path).map_or(0, Vec::len)
        } else {
            0
        };
        Ok(Box::new(MemoryFile {
            path: path.to_vec(),
            contents: Arc::clone(&self.contents),
            position,
            readable: kind == b'r' || plus,
            writable: kind != b'r' || plus,
            append: kind == b'a',
        }))
    }

    fn remove(&mut self, path: &[u8]) -> Result<(), HostError> {
        match locked(&self.contents).remove(path) {
            Some(_) => Ok(()),
            None => Err(HostError::not_found()),
        }
    }

    fn rename(&mut self, from: &[u8], to: &[u8]) -> Result<(), HostError> {
        let mut contents = locked(&self.contents);
        let Some(data) = contents.remove(from) else {
            return Err(HostError::not_found());
        };
        contents.insert(to.to_vec(), data);
        Ok(())
    }

    fn tmpname(&mut self) -> Result<Vec<u8>, HostError> {
        let mut counter = locked(&self.counter);
        *counter += 1;
        Ok(format!("/tmp/lua_{:06x}", *counter).into_bytes())
    }
}

impl HostFile for MemoryFile {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, HostError> {
        if !self.readable {
            return Err(HostError::new(EBADF, "Bad file descriptor"));
        }
        let contents = locked(&self.contents);
        let data = contents.get(&self.path).map_or(&[][..], Vec::as_slice);
        let available = data.get(self.position..).unwrap_or(&[]);
        let count = available.len().min(buffer.len());
        buffer[..count].copy_from_slice(&available[..count]);
        self.position += count;
        Ok(count)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        if !self.writable {
            return Err(HostError::new(EBADF, "Bad file descriptor"));
        }
        let mut contents = locked(&self.contents);
        let data = contents.entry(self.path.clone()).or_default();
        if self.append {
            self.position = data.len();
        }
        let end = self.position + bytes.len();
        if data.len() < end {
            data.resize(end, 0);
        }
        data[self.position..end].copy_from_slice(bytes);
        self.position = end;
        Ok(())
    }

    fn seek(&mut self, position: Seek) -> Result<u64, HostError> {
        let length = locked(&self.contents).get(&self.path).map_or(0, Vec::len) as i64;
        let target = match position {
            Seek::Start(offset) => offset as i64,
            Seek::Current(offset) => self.position as i64 + offset,
            Seek::End(offset) => length + offset,
        };
        if target < 0 {
            return Err(HostError::new(EINVAL, "Invalid argument"));
        }
        self.position = target as usize;
        Ok(target as u64)
    }

    fn flush(&mut self) -> Result<(), HostError> {
        Ok(())
    }
}

/// The clock always reads the same, so `os.time` and `os.clock` repeat.
#[derive(Default)]
pub struct FixedSystem;

impl System for FixedSystem {
    fn now(&mut self) -> i64 {
        0
    }

    fn processor_seconds(&mut self) -> f64 {
        0.0
    }

    fn getenv(&mut self, _name: &[u8]) -> Option<Vec<u8>> {
        None
    }

    fn break_down(&mut self, seconds: i64, _utc: bool) -> Option<TimeFields> {
        civil_fields(seconds)
    }

    fn exit(&mut self, _code: i32) {}

    fn executable_directory(&mut self) -> Option<Vec<u8>> {
        None
    }
}

pub fn civil_fields(seconds: i64) -> Option<TimeFields> {
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let yearday = days - days_from_civil(year, 1, 1) + 1;
    Some(TimeFields {
        year,
        month,
        day,
        hour: day_seconds / 3600,
        minute: day_seconds % 3600 / 60,
        second: day_seconds % 60,
        weekday: (days + 4).rem_euclid(7) + 1,
        yearday,
        daylight_saving: Some(false),
    })
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let month_index = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}
