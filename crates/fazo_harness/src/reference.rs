use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{ChildStdin, Command, ExitStatus, Stdio},
    sync::{
        OnceLock,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

pub const VERSION: &str = "5.5.1";
pub const SOURCE_URL: &str = "https://www.lua.org/ftp/lua-5.5.1.tar.gz";
pub const SOURCE_SHA256: &str = "1c4b4068d67061f2a2231ad2b5422e77acea1487ea9890f6320af614f4373dce";
pub const SUITE_URL: &str = "https://www.lua.org/tests/lua-5.5.1-tests.tar.gz";
pub const SUITE_SHA256: &str = "da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31";

pub const SOURCE_DIR: &str = "lua-5.5.1";
const SUITE_DIR: &str = "lua-5.5.1-tests";
const CACHE_NAME: &str = "fazo-reference";
const EXE_SUFFIX: &str = std::env::consts::EXE_SUFFIX;
const WORK_DIR: &str = "work";
const LUA_MAIN: &str = "lua.c";
const LUAC_MAIN: &str = "luac.c";
const SUITE_LIBS_DIR: &str = "libs";
#[cfg(target_os = "linux")]
const SUITE_LIBRARIES: &[(&str, &str)] = &[
    ("lib1.so", "lib1.c"),
    ("lib11.so", "lib11.c"),
    ("lib2.so", "lib2.c"),
    ("lib21.so", "lib21.c"),
    ("lib2-v2.so", "lib22.c"),
];

pub struct Reference {
    pub lua: PathBuf,
    pub luac: PathBuf,
    pub suite: PathBuf,
}

pub struct Captured {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub success: bool,
    pub code: Option<i32>,
}

/// `FAZO_REFERENCE_DIR` overrides the default, which sits in the cargo target directory.
pub fn cache_dir() -> PathBuf {
    let path = match env::var_os("FAZO_REFERENCE_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => target_dir().join(CACHE_NAME),
    };
    std::path::absolute(&path).unwrap_or(path)
}

fn target_dir() -> PathBuf {
    env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .or_else(running_target_dir)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("target"))
}

fn running_target_dir() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    exe.ancestors()
        .find(|dir| dir.join("CACHEDIR.TAG").is_file())
        .map(Path::to_path_buf)
}

static TIMEOUT_SCALE: AtomicU32 = AtomicU32::new(1);

pub fn set_timeout_scale(scale: u32) {
    TIMEOUT_SCALE.store(scale, Ordering::Relaxed);
}

pub fn scaled(deadline: Duration) -> Duration {
    deadline * TIMEOUT_SCALE.load(Ordering::Relaxed)
}

static SHARED: OnceLock<Result<Reference, String>> = OnceLock::new();

/// The reference for this process: `ensure` runs once, and later calls reuse its result.
pub fn shared() -> Result<&'static Reference, String> {
    SHARED
        .get_or_init(ensure)
        .as_ref()
        .map_err(|error| {
            format!(
                "{error}\nthe reference lives in {}; with network access, \
                 `cargo run --manifest-path crates/fazo_harness/Cargo.toml --bin scoreboard -- fetch` \
                 populates it",
                cache_dir().display()
            )
        })
}

pub fn ensure() -> Result<Reference, String> {
    let root = cache_dir();
    let bin = root.join(format!("bin-{VERSION}"));
    let lua = bin.join(format!("lua{EXE_SUFFIX}"));
    let luac = bin.join(format!("luac{EXE_SUFFIX}"));
    if !root.join(SOURCE_DIR).is_dir() || !root.join(SUITE_DIR).is_dir() {
        fetch_into(&root)?;
    }
    let src = root.join(SOURCE_DIR).join("src");
    if !lua.is_file() {
        build(&src, &lua, LUA_MAIN)?;
    }
    if !luac.is_file() {
        build(&src, &luac, LUAC_MAIN)?;
    }
    build_suite_libraries(&src, &root.join(SUITE_DIR).join(SUITE_LIBS_DIR))?;
    Ok(Reference {
        lua,
        luac,
        suite: root.join(SUITE_DIR),
    })
}

pub fn fetch_into(root: &Path) -> Result<(), String> {
    fs::create_dir_all(root).map_err(|error| format!("cannot create {}: {error}", root.display()))?;
    if !root.join(SOURCE_DIR).is_dir() {
        let bytes = download(SOURCE_URL, SOURCE_SHA256)?;
        extract(&bytes, root)?;
    }
    if !root.join(SUITE_DIR).is_dir() {
        let bytes = download(SUITE_URL, SUITE_SHA256)?;
        extract(&bytes, root)?;
    }
    Ok(())
}

fn download(url: &str, expected: &str) -> Result<Vec<u8>, String> {
    let mut response = ureq::get(url)
        .call()
        .map_err(|error| format!("cannot download {url}: {error}"))?;
    let bytes = response
        .body_mut()
        .read_to_vec()
        .map_err(|error| format!("cannot read {url}: {error}"))?;
    let actual = sha256_hex(&bytes);
    if actual != expected {
        return Err(format!(
            "{url} has SHA-256 {actual}, expected {expected}; refusing to use it"
        ));
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn extract(bytes: &[u8], root: &Path) -> Result<(), String> {
    let decoder = flate2::read::GzDecoder::new(bytes);
    tar::Archive::new(decoder)
        .unpack(root)
        .map_err(|error| format!("cannot extract into {}: {error}", root.display()))
}

fn build(src: &Path, exe: &Path, main: &str) -> Result<(), String> {
    let sources = c_sources(src, main)?;
    let out_dir = exe
        .parent()
        .ok_or_else(|| format!("{} has no folder", exe.display()))?;
    fs::create_dir_all(out_dir).map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;
    let tool = c_compiler()?;
    let mut command = tool.to_command();
    command.current_dir(out_dir).stdin(Stdio::null());
    if tool.is_like_msvc() {
        msvc_arguments(&mut command, &sources, exe, out_dir);
    } else {
        gnu_arguments(&mut command, &sources, exe);
    }
    run_compiler(&mut command, "lua")
}

fn c_compiler() -> Result<cc::Tool, String> {
    let triple = host_triple();
    cc::Build::new()
        .target(&triple)
        .host(&triple)
        .opt_level(2)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .try_get_compiler()
        .map_err(|error| format!("no C compiler for {triple}: {error}"))
}

fn run_compiler(command: &mut Command, product: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("cannot run the C compiler: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "building {product} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// Built with the suite's `libs/makefile` flags, which produce Linux shared objects only.
#[cfg(target_os = "linux")]
fn build_suite_libraries(src: &Path, libs: &Path) -> Result<(), String> {
    for (library, source) in SUITE_LIBRARIES {
        let product = libs.join(library);
        if product.is_file() {
            continue;
        }
        let mut command = c_compiler()?.to_command();
        command.current_dir(libs).stdin(Stdio::null());
        command.args(["-Wall", "-O2"]).arg("-I").arg(src).args(["-fPIC", "-shared", "-o"]);
        command.arg(&product).arg(libs.join(source));
        run_compiler(&mut command, library)?;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn build_suite_libraries(_src: &Path, _libs: &Path) -> Result<(), String> {
    Ok(())
}

fn c_sources(src: &Path, main: &str) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(src).map_err(|error| format!("cannot read {}: {error}", src.display()))?;
    let mut sources = Vec::new();
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        let is_c = path.extension().is_some_and(|extension| extension == "c");
        let is_other_main = path
            .file_name()
            .is_some_and(|name| name != main && (name == LUA_MAIN || name == LUAC_MAIN));
        if is_c && !is_other_main {
            sources.push(path);
        }
    }
    sources.sort();
    Ok(sources)
}

fn platform_define() -> Option<&'static str> {
    if cfg!(target_os = "linux") {
        Some("-DLUA_USE_LINUX")
    } else if cfg!(target_os = "macos") {
        Some("-DLUA_USE_MACOSX")
    } else {
        None
    }
}

fn gnu_arguments(command: &mut Command, sources: &[PathBuf], exe: &Path) {
    command.args(["-std=gnu99", "-O2", "-Wall", "-Wextra"]);
    command.args(platform_define());
    command.args(sources);
    command.arg("-o").arg(exe);
    if cfg!(target_os = "linux") {
        command.args(["-Wl,-E", "-ldl"]);
    }
    if cfg!(target_os = "macos") {
        command.arg("-lreadline");
    }
    command.arg("-lm");
}

fn msvc_arguments(command: &mut Command, sources: &[PathBuf], exe: &Path, out_dir: &Path) {
    command.args(["/nologo", "/O2", "/W3"]);
    let mut executable = std::ffi::OsString::from("/Fe");
    executable.push(exe);
    command.arg(executable);
    let mut objects = std::ffi::OsString::from("/Fo");
    objects.push(out_dir);
    objects.push(std::path::MAIN_SEPARATOR.to_string());
    command.arg(objects);
    command.args(sources);
}

fn host_triple() -> String {
    let arch = env::consts::ARCH;
    let rest = if cfg!(target_os = "windows") {
        if cfg!(target_env = "msvc") { "pc-windows-msvc" } else { "pc-windows-gnu" }
    } else if cfg!(target_os = "macos") {
        "apple-darwin"
    } else {
        "unknown-linux-gnu"
    };
    format!("{arch}-{rest}")
}

pub fn execute(lua: &Path, args: &[&str], dir: &Path, deadline: Duration) -> Result<Captured, String> {
    run_process(lua, args, &[], &[], dir, deadline)
}

/// The init variables are removed first, so no run picks up the developer's own init script.
pub fn run_process(
    program: &Path,
    args: &[&str],
    stdin: &[u8],
    env: &[(&str, &str)],
    dir: &Path,
    deadline: Duration,
) -> Result<Captured, String> {
    let mut child = Command::new(program)
        .args(args)
        .current_dir(dir)
        .env_remove("LUA_INIT")
        .env_remove("LUA_INIT_5_5")
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", program.display()))?;
    let feeder = write_in_background(child.stdin.take(), stdin.to_vec());
    let stdout = read_in_background(child.stdout.take());
    let stderr = read_in_background(child.stderr.take());
    let status = wait_until(&mut child, deadline)?;
    let _ = feeder.join();
    Ok(Captured {
        stdout: text_mode_newlines(stdout.join().unwrap_or_default()),
        stderr: text_mode_newlines(stderr.join().unwrap_or_default()),
        success: status.success(),
        code: status.code(),
    })
}

fn write_in_background(pipe: Option<ChildStdin>, bytes: Vec<u8>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Some(mut pipe) = pipe {
            let _ = pipe.write_all(&bytes);
        }
    })
}

fn text_mode_newlines(bytes: Vec<u8>) -> Vec<u8> {
    if !cfg!(windows) {
        return bytes;
    }
    let mut text = Vec::with_capacity(bytes.len());
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            continue;
        }
        text.push(byte);
    }
    text
}

fn read_in_background<R: Read + Send + 'static>(pipe: Option<R>) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        bytes
    })
}

fn wait_until(child: &mut std::process::Child, deadline: Duration) -> Result<ExitStatus, String> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Ok(status);
        }
        if started.elapsed() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timeout after {} seconds", deadline.as_secs()));
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Skips the program name the standalone interpreter prints first; it is the path the harness started.
pub fn error_message(lua: &Path, captured: &Captured) -> Vec<u8> {
    let first = first_line(&captured.stderr);
    let by_path = first.strip_prefix(format!("{}: ", lua.display()).as_bytes());
    by_path.or_else(|| first.strip_prefix(b"lua: ")).unwrap_or(first).to_vec()
}

pub fn first_line(bytes: &[u8]) -> &[u8] {
    let line = bytes.split(|&byte| byte == b'\n').next().unwrap_or_default();
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// A unique folder under the cache's work directory, removed when dropped.
pub struct WorkDir {
    path: PathBuf,
}

static NEXT_WORK_DIR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl WorkDir {
    pub fn create() -> Result<Self, String> {
        let id = NEXT_WORK_DIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = cache_dir()
            .join(WORK_DIR)
            .join(format!("{}-{id}", std::process::id()));
        fs::create_dir_all(&path).map_err(|error| format!("cannot create {}: {error}", path.display()))?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
