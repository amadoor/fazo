use std::{
    borrow::Cow,
    ffi::{OsStr, OsString},
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek as _, SeekFrom, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use chrono::{Datelike, Local, NaiveDateTime, TimeZone, Timelike, Utc};
use fazo::{ExitStatus, FileSystem, Host, HostError, HostFile, Seek, Stdio as VmStdio, System, TimeFields};

const ENOENT: i32 = 2;
const EACCES: i32 = 13;
const EEXIST: i32 = 17;
const EBADF: i32 = 9;
const EINVAL: i32 = 22;
const ESPIPE: i32 = 29;

#[cfg(unix)]
pub fn argument_bytes(argument: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    argument.as_bytes().to_vec()
}

#[cfg(windows)]
pub fn argument_bytes(argument: &OsStr) -> Vec<u8> {
    argument.to_string_lossy().into_owned().into_bytes()
}

#[cfg(unix)]
pub fn path_of(name: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    PathBuf::from(OsStr::from_bytes(name))
}

#[cfg(windows)]
pub fn path_of(name: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(name).into_owned())
}

pub fn host() -> Host {
    Host::new(
        Box::new(ProcessStdio::default()),
        Box::new(ProcessFiles),
        Box::new(ProcessSystem::new()),
    )
}

const TEXT_NEWLINES: bool = cfg!(windows);

pub fn to_text_output(bytes: &[u8]) -> Cow<'_, [u8]> {
    if !TEXT_NEWLINES || !bytes.contains(&b'\n') {
        return Cow::Borrowed(bytes);
    }
    let mut out = Vec::with_capacity(bytes.len() + 16);
    for &byte in bytes {
        if byte == b'\n' {
            out.push(b'\r');
        }
        out.push(byte);
    }
    Cow::Owned(out)
}

#[derive(Default)]
struct TextInput {
    held: Option<u8>,
}

impl TextInput {
    fn read(&mut self, source: &mut dyn Read, buffer: &mut [u8]) -> io::Result<usize> {
        let mut written = 0;
        while written < buffer.len() {
            let Some(byte) = self.next_byte(source)? else {
                break;
            };
            if !TEXT_NEWLINES || byte != b'\r' {
                buffer[written] = byte;
                written += 1;
                continue;
            }
            match self.next_byte(source)? {
                Some(b'\n') => buffer[written] = b'\n',
                Some(following) => {
                    buffer[written] = b'\r';
                    self.held = Some(following);
                }
                None => buffer[written] = b'\r',
            }
            written += 1;
        }
        Ok(written)
    }

    fn next_byte(&mut self, source: &mut dyn Read) -> io::Result<Option<u8>> {
        if let Some(byte) = self.held.take() {
            return Ok(Some(byte));
        }
        let mut one = [0u8; 1];
        Ok((source.read(&mut one)? == 1).then_some(one[0]))
    }
}

#[derive(Default)]
struct ProcessStdio {
    input: TextInput,
}

impl VmStdio for ProcessStdio {
    fn write_stdout(&mut self, bytes: &[u8]) {
        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(&to_text_output(bytes));
        let _ = stdout.flush();
    }

    fn write_stderr(&mut self, bytes: &[u8]) {
        let _ = io::stderr().lock().write_all(&to_text_output(bytes));
    }

    fn read_stdin(&mut self, buffer: &mut [u8]) -> usize {
        self.input.read(&mut io::stdin().lock(), buffer).unwrap_or(0)
    }
}

struct ProcessFiles;

impl FileSystem for ProcessFiles {
    fn open(&mut self, path: &[u8], mode: &[u8]) -> Result<Box<dyn HostFile>, HostError> {
        let file = open_file(&path_of(path), mode).map_err(|error| host_error(&error))?;
        let text = TEXT_NEWLINES && !mode.contains(&b'b');
        Ok(Box::new(ProcessFile {
            file,
            text,
            input: TextInput::default(),
        }))
    }

    fn remove(&mut self, path: &[u8]) -> Result<(), HostError> {
        fs::remove_file(path_of(path)).map_err(|error| host_error(&error))
    }

    fn rename(&mut self, from: &[u8], to: &[u8]) -> Result<(), HostError> {
        fs::rename(path_of(from), path_of(to)).map_err(|error| host_error(&error))
    }

    #[cfg(unix)]
    fn tmpname(&mut self) -> Result<Vec<u8>, HostError> {
        const ATTEMPTS: usize = 100;
        let mut random = Random::new();
        for _ in 0..ATTEMPTS {
            let name = format!("/tmp/lua_{}", random.letters(6));
            match OpenOptions::new().write(true).create_new(true).open(&name) {
                Ok(_) => return Ok(name.into_bytes()),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(host_error(&error)),
            }
        }
        Err(HostError::new(EEXIST, "File exists"))
    }

    #[cfg(windows)]
    fn tmpname(&mut self) -> Result<Vec<u8>, HostError> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = std::env::temp_dir().join(format!("lua_{}_{serial}", std::process::id()));
        Ok(name.to_string_lossy().into_owned().into_bytes())
    }

    fn popen(&mut self, command: &[u8], mode: &[u8]) -> Result<Box<dyn HostFile>, HostError> {
        let mut process = shell_command(command);
        if mode == b"r" {
            process.stdout(Stdio::piped());
        } else {
            process.stdin(Stdio::piped());
        }
        let mut child = process.spawn().map_err(|error| host_error(&error))?;
        Ok(Box::new(PipeFile {
            stdin: child.stdin.take(),
            stdout: child.stdout.take(),
            child: Some(child),
            input: TextInput::default(),
            text: TEXT_NEWLINES,
        }))
    }
}

#[cfg(unix)]
fn shell_command(command: &[u8]) -> Command {
    use std::os::unix::ffi::OsStrExt;
    let mut process = Command::new("sh");
    process.arg("-c").arg(OsStr::from_bytes(command));
    process
}

#[cfg(windows)]
fn shell_command(command: &[u8]) -> Command {
    use std::os::windows::process::CommandExt;
    let mut process = Command::new("cmd");
    process.raw_arg(format!("/c {}", String::from_utf8_lossy(command)));
    process
}

#[cfg(unix)]
fn exit_status_of(status: std::process::ExitStatus) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => ExitStatus::Exited(code),
        (None, Some(signal)) => ExitStatus::Signaled(signal),
        (None, None) => ExitStatus::Exited(-1),
    }
}

#[cfg(windows)]
fn exit_status_of(status: std::process::ExitStatus) -> ExitStatus {
    ExitStatus::Exited(status.code().unwrap_or(-1))
}

#[cfg(unix)]
struct Random {
    state: u64,
}

#[cfg(unix)]
impl Random {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |time| time.as_nanos() as u64);
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        let seed = nanos ^ (u64::from(std::process::id()) << 32) ^ serial.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Self { state: seed | 1 }
    }

    fn letters(&mut self, count: usize) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        (0..count)
            .map(|_| {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                char::from(ALPHABET[(self.state % ALPHABET.len() as u64) as usize])
            })
            .collect()
    }
}

fn open_file(path: &PathBuf, mode: &[u8]) -> io::Result<File> {
    let plus = mode.contains(&b'+');
    let mut options = OpenOptions::new();
    match mode.first() {
        Some(b'w') => options.write(true).create(true).truncate(true).read(plus),
        Some(b'a') => options.append(true).create(true).read(plus),
        _ => options.read(true).write(plus),
    };
    options.open(path)
}

struct ProcessFile {
    file: File,
    text: bool,
    input: TextInput,
}

impl HostFile for ProcessFile {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, HostError> {
        if !self.text {
            return self.file.read(buffer).map_err(|error| host_error(&error));
        }
        self.input.read(&mut self.file, buffer).map_err(|error| host_error(&error))
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        let bytes = if self.text { to_text_output(bytes) } else { Cow::Borrowed(bytes) };
        self.file.write_all(&bytes).map_err(|error| host_error(&error))
    }

    fn seek(&mut self, position: Seek) -> Result<u64, HostError> {
        let target = match position {
            Seek::Start(offset) => SeekFrom::Start(offset),
            Seek::Current(offset) => SeekFrom::Current(offset),
            Seek::End(offset) => SeekFrom::End(offset),
        };
        self.input = TextInput::default();
        self.file.seek(target).map_err(|error| host_error(&error))
    }

    fn flush(&mut self) -> Result<(), HostError> {
        self.file.flush().map_err(|error| host_error(&error))
    }
}

struct PipeFile {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    input: TextInput,
    text: bool,
}

impl HostFile for PipeFile {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, HostError> {
        let Some(stdout) = self.stdout.as_mut() else {
            return Err(HostError::new(EBADF, "Bad file descriptor"));
        };
        if !self.text {
            return stdout.read(buffer).map_err(|error| host_error(&error));
        }
        self.input.read(stdout, buffer).map_err(|error| host_error(&error))
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), HostError> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(HostError::new(EBADF, "Bad file descriptor"));
        };
        let bytes = if self.text { to_text_output(bytes) } else { Cow::Borrowed(bytes) };
        stdin.write_all(&bytes).map_err(|error| host_error(&error))
    }

    fn seek(&mut self, _position: Seek) -> Result<u64, HostError> {
        Err(HostError::new(ESPIPE, "Illegal seek"))
    }

    fn flush(&mut self) -> Result<(), HostError> {
        match self.stdin.as_mut() {
            Some(stdin) => stdin.flush().map_err(|error| host_error(&error)),
            None => Ok(()),
        }
    }

    fn finish(&mut self) -> Result<Option<ExitStatus>, HostError> {
        self.stdin = None;
        self.stdout = None;
        let Some(mut child) = self.child.take() else {
            return Ok(None);
        };
        let status = child.wait().map_err(|error| host_error(&error))?;
        Ok(Some(exit_status_of(status)))
    }
}

fn host_error(error: &io::Error) -> HostError {
    let (errno, reason) = match error.kind() {
        io::ErrorKind::NotFound => (ENOENT, "No such file or directory".to_string()),
        io::ErrorKind::PermissionDenied => (EACCES, "Permission denied".to_string()),
        io::ErrorKind::AlreadyExists => (EEXIST, "File exists".to_string()),
        io::ErrorKind::InvalidInput => (EINVAL, "Invalid argument".to_string()),
        _ => {
            let text = error.to_string();
            let reason = match text.rfind(" (os error ") {
                Some(at) => text[..at].to_string(),
                None => text,
            };
            (error.raw_os_error().unwrap_or(0), reason)
        }
    };
    HostError { errno, reason }
}

struct ProcessSystem {
    started: Instant,
}

impl ProcessSystem {
    fn new() -> Self {
        Self { started: Instant::now() }
    }
}

impl System for ProcessSystem {
    fn now(&mut self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs() as i64)
    }

    fn processor_seconds(&mut self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn getenv(&mut self, name: &[u8]) -> Option<Vec<u8>> {
        let key: OsString = String::from_utf8_lossy(name).into_owned().into();
        std::env::var_os(key).map(|value| argument_bytes(&value))
    }

    fn break_down(&mut self, seconds: i64, utc: bool) -> Option<TimeFields> {
        if utc {
            let moment = Utc.timestamp_opt(seconds, 0).single()?;
            return Some(fields_of(moment.naive_utc(), Some(false)));
        }
        let moment = Local.timestamp_opt(seconds, 0).single()?;
        Some(fields_of(moment.naive_local(), None))
    }

    fn exit(&mut self, code: i32) {
        std::process::exit(code);
    }

    fn executable_directory(&mut self) -> Option<Vec<u8>> {
        let executable = std::env::current_exe().ok()?;
        let directory = executable.parent()?;
        Some(argument_bytes(directory.as_os_str()))
    }

    fn shell_available(&mut self) -> bool {
        true
    }

    fn execute(&mut self, command: &[u8]) -> Result<ExitStatus, HostError> {
        let status = shell_command(command).status().map_err(|error| host_error(&error))?;
        Ok(exit_status_of(status))
    }
}

fn fields_of(moment: NaiveDateTime, daylight_saving: Option<bool>) -> TimeFields {
    TimeFields {
        year: i64::from(moment.year()),
        month: i64::from(moment.month()),
        day: i64::from(moment.day()),
        hour: i64::from(moment.hour()),
        minute: i64::from(moment.minute()),
        second: i64::from(moment.second()),
        weekday: i64::from(moment.weekday().num_days_from_sunday()) + 1,
        yearday: i64::from(moment.ordinal()),
        daylight_saving,
    }
}
