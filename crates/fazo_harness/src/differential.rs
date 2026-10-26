use std::{
    fs,
    panic::{self, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU8, Ordering},
        mpsc::{self, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use fazo::{Options, RequestHandle, Vm};

use crate::reference::{self, WorkDir, first_line};

pub const CHUNK_NAME: &str = "t";

pub const CASE_DEADLINE: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub output: Vec<u8>,
    pub error: Option<Vec<u8>>,
    pub timed_out: bool,
}

impl Outcome {
    pub fn normalized(self, chunk_name: &str) -> Self {
        Self {
            output: normalize_nan_text(&self.output),
            error: self.error.map(|text| normalize_nan_text(&normalize_error(&text, chunk_name))),
            timed_out: self.timed_out,
        }
    }

    pub fn render(&self) -> String {
        let output = String::from_utf8_lossy(&self.output);
        match &self.error {
            Some(error) => format!("{output}error: {}\n", String::from_utf8_lossy(error)),
            None => output.into_owned(),
        }
    }
}

#[derive(Debug)]
pub struct Comparison {
    pub fazo: Outcome,
    pub reference: Outcome,
}

impl Comparison {
    pub fn matches(&self) -> bool {
        self.fazo == self.reference
    }

    pub fn timed_out(&self) -> bool {
        self.fazo.timed_out
    }
}

pub fn compare(source: &[u8]) -> Result<Comparison, String> {
    let reference = run_reference(source, CHUNK_NAME)?.normalized(CHUNK_NAME);
    let fazo = run_fazo_with_standard_libraries(source, CHUNK_NAME).normalized(CHUNK_NAME);
    Ok(Comparison { fazo, reference })
}

/// Runs `source` as `chunk_name` with the official interpreter; the error is its first stderr line without `lua: `.
pub fn run_reference(source: &[u8], chunk_name: &str) -> Result<Outcome, String> {
    let lua = reference::shared()?;
    let work = WorkDir::create()?;
    fs::write(work.path().join(chunk_name), source)
        .map_err(|error| format!("cannot write the script: {error}"))?;
    let outcome = match reference::execute(&lua.lua, &[chunk_name], work.path(), reference::scaled(CASE_DEADLINE)) {
        Ok(captured) => Outcome {
            output: captured.stdout.clone(),
            error: (!captured.success).then(|| reference::error_message(&lua.lua, &captured)),
            timed_out: false,
        },
        Err(error) => Outcome {
            output: Vec::new(),
            error: Some(error.into_bytes()),
            timed_out: false,
        },
    };
    Ok(outcome)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitMode {
    Normal,
    Interpreter,
    Baseline,
    Peak,
    Optimizing,
}

impl JitMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "normal" => Some(Self::Normal),
            "interpreter" => Some(Self::Interpreter),
            "baseline" => Some(Self::Baseline),
            "peak" => Some(Self::Peak),
            "optimizing" => Some(Self::Optimizing),
            _ => None,
        }
    }

    pub fn flags(self) -> &'static [&'static str] {
        match self {
            Self::Normal => &[],
            Self::Interpreter => &["-X", "tier=interpreter"],
            Self::Baseline => &["-X", "force=baseline"],
            Self::Peak => &["-X", "force=peak"],
            Self::Optimizing => &["-X", "force=optimizing"],
        }
    }
}

static JIT_MODE: AtomicU8 = AtomicU8::new(0);

/// Applies to every fazo run the harness starts after this call.
pub fn set_jit_mode(mode: JitMode) {
    JIT_MODE.store(mode as u8, Ordering::Relaxed);
}

pub fn jit_mode() -> JitMode {
    match JIT_MODE.load(Ordering::Relaxed) {
        1 => JitMode::Interpreter,
        2 => JitMode::Baseline,
        3 => JitMode::Peak,
        4 => JitMode::Optimizing,
        _ => JitMode::Normal,
    }
}

fn fazo_options() -> Options {
    let mut options = Options::default();
    for option in jit_mode().flags().iter().skip(1).step_by(2) {
        options.set_extension(option.as_bytes());
    }
    options
}

pub fn run_fazo(source: &[u8], chunk_name: &str) -> Outcome {
    run_fazo_with(fazo_options(), source, chunk_name)
}

/// Opens `os`, `io` and `package`, which the official test files need.
pub fn run_fazo_with_standard_libraries(source: &[u8], chunk_name: &str) -> Outcome {
    let mut options = fazo_options();
    options.os = true;
    options.io = true;
    options.package = true;
    run_fazo_with(options, source, chunk_name)
}

fn run_fazo_with(options: Options, source: &[u8], chunk_name: &str) -> Outcome {
    run_fazo_until(options, source, chunk_name, reference::scaled(CASE_DEADLINE))
}

fn run_fazo_until(options: Options, source: &[u8], chunk_name: &str, deadline: Duration) -> Outcome {
    let mut vm = Vm::new(options);
    vm.use_generational_collector();
    let chunkname = format!("@{chunk_name}");
    let timer = Deadline::start(vm.interrupt_handle(), deadline);

    let attempt = panic::catch_unwind(AssertUnwindSafe(|| vm.run_script(chunkname.as_bytes(), source)));
    let expired = timer.finish();
    let timed_out = expired && matches!(&attempt, Ok(Err(error)) if error.is_interrupted());
    let error = match attempt {
        _ if timed_out => Some(format!("timeout after {} seconds", deadline.as_secs()).into_bytes()),
        Ok(result) => result.err().map(|error| first_line(&vm.error_message(&error)).to_vec()),
        Err(panic) => Some(format!("PANIC: {}", panic_text(panic.as_ref())).into_bytes()),
    };
    Outcome {
        output: vm.take_output(),
        error,
        timed_out,
    }
}

/// Interrupts the VM from a timer thread once the deadline passes; finishing first stops the timer.
struct Deadline {
    stop: Sender<()>,
    timer: JoinHandle<bool>,
}

impl Deadline {
    fn start(interrupt: RequestHandle, deadline: Duration) -> Self {
        let (stop, stopped) = mpsc::channel::<()>();
        let timer = thread::spawn(move || match stopped.recv_timeout(deadline) {
            Err(RecvTimeoutError::Timeout) => {
                interrupt.request();
                true
            }
            Ok(()) | Err(RecvTimeoutError::Disconnected) => false,
        });
        Self { stop, timer }
    }

    fn finish(self) -> bool {
        let _ = self.stop.send(());
        self.timer.join().unwrap_or(false)
    }
}

/// Strips `chunk:line: ` so messages compare by text; positions compare only where a case prints them on purpose.
pub fn normalize_error(text: &[u8], chunk_name: &str) -> Vec<u8> {
    let Some(rest) = text.strip_prefix(format!("{chunk_name}:").as_bytes()) else {
        return text.to_vec();
    };
    let Some(split) = rest.windows(2).position(|pair| pair == b": ") else {
        return text.to_vec();
    };
    let (line, message) = (&rest[..split], &rest[split + 2..]);
    if line.is_empty() || !line.iter().all(u8::is_ascii_digit) {
        return text.to_vec();
    }
    message.to_vec()
}

/// C libraries disagree on NaN sign and payload text, and fazo prints every NaN as `nan`.
pub fn normalize_nan_text(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        match nan_spelling_end(text, index) {
            Some((end, upper)) => {
                out.extend_from_slice(if upper { b"NAN" } else { b"nan" });
                index = end;
            }
            None => {
                out.push(text[index]);
                index += 1;
            }
        }
    }
    out
}

fn nan_spelling_end(text: &[u8], start: usize) -> Option<(usize, bool)> {
    let is_word = |byte: &u8| byte.is_ascii_alphanumeric() || *byte == b'_';
    if start > 0 && is_word(&text[start - 1]) {
        return None;
    }
    let mut index = start + usize::from(text.get(start) == Some(&b'-'));
    let upper = match text.get(index..index + 3)? {
        b"nan" => false,
        b"NAN" => true,
        _ => return None,
    };
    index += 3;
    let suffixes: [&[u8]; 2] = if upper { [b"(IND)", b"(SNAN)"] } else { [b"(ind)", b"(snan)"] };
    if let Some(suffix) = suffixes.iter().find(|suffix| text[index..].starts_with(suffix)) {
        index += suffix.len();
    }
    match text.get(index) {
        Some(byte) if is_word(byte) => None,
        _ => Some((index, upper)),
    }
}

fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()))
        .unwrap_or_else(|| "non-string panic".to_string())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{CHUNK_NAME, fazo_options, normalize_error, run_fazo_until, run_reference};

    #[test]
    fn an_in_process_run_past_its_deadline_is_interrupted_and_timed_out() {
        let outcome = run_fazo_until(fazo_options(), b"print('started') while true do end", CHUNK_NAME, Duration::from_millis(50));
        assert!(outcome.timed_out);
        assert_eq!(outcome.output, b"started
");
        assert!(outcome.error.is_some_and(|error| error.starts_with(b"timeout after")));

        let finished = run_fazo_until(fazo_options(), b"print(1)", CHUNK_NAME, Duration::from_secs(60));
        assert!(!finished.timed_out);
        assert_eq!(finished.error, None);
    }

    #[test]
    fn position_prefix_is_stripped_only_for_its_chunk() {
        assert_eq!(normalize_error(b"t:12: boom", CHUNK_NAME), b"boom");
        assert_eq!(normalize_error(b"t:x: boom", CHUNK_NAME), b"t:x: boom");
        assert_eq!(normalize_error(b"other:3: boom", CHUNK_NAME), b"other:3: boom");
        assert_eq!(normalize_error(b"boom", CHUNK_NAME), b"boom");
    }

    #[test]
    fn non_utf8_bytes_are_compared_as_bytes() {
        assert_eq!(normalize_error(b"t:1: \xff", CHUNK_NAME), b"\xff");
        assert_ne!(normalize_error(b"t:1: \xff", CHUNK_NAME), normalize_error(b"t:1: \xfe", CHUNK_NAME));
    }

    #[test]
    fn reference_print_writes_tab_separated_tostring_lines() {
        let outcome = run_reference(b"print(1, 2.0, 'x', nil, -0.0)\nprint()", CHUNK_NAME)
            .expect("the reference is cached; see reference::shared for how to populate it");
        assert_eq!(outcome.output, b"1\t2.0\tx\tnil\t-0.0\n\n");
        assert_eq!(outcome.error, None);
    }

    #[test]
    fn reference_error_message_drops_the_lua_prefix_and_traceback() {
        let outcome = run_reference(b"error('boom')", CHUNK_NAME)
            .expect("the reference is cached; see reference::shared for how to populate it");
        assert_eq!(outcome.error.as_deref(), Some(&b"t:1: boom"[..]));
    }

    #[test]
    fn nan_spellings_normalize_to_nan_and_keep_other_words() {
        use super::normalize_nan_text;
        assert_eq!(normalize_nan_text(b"nan -nan -nan(ind) nan(snan)"), b"nan nan nan nan");
        assert_eq!(normalize_nan_text(b"NAN -NAN(IND)"), b"NAN NAN");
        assert_eq!(normalize_nan_text(b"nano banana_nan inf -inf 1.5"), b"nano banana_nan inf -inf 1.5");
        assert_eq!(normalize_nan_text(b"nan(x)"), b"nan(x)");
    }
}
