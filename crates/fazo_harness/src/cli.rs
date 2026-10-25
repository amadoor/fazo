
use std::{fs, path::Path, time::Duration};

use crate::{
    differential::normalize_nan_text,
    reference::{self, WorkDir},
};

const CASE_DEADLINE: Duration = Duration::from_secs(120);
const BANNER_MARKS: &[&[u8]] = &[b"Lua.org, PUC-Rio", b"github.com/amadoor/fazo"];
const TRACEBACK_HEADER: &[u8] = b"stack traceback:\n";
const PROGRAM_NAME: &[u8] = b"fazo";
const NORMALIZED_PROGRAM: &[u8] = b"<prog>";
const NORMALIZED_DIRECTORY: &[u8] = b"<dir>";

const DEPTH_PROGRAM: &str = "local n = 0\nlocal function f() n = n + 1; io.write(n, \"\\n\"); return 1 + f() end\nf()\n";
const OVERFLOW_PROGRAM: &str = "local function f() return 1 + f() end\nf()\n";

pub struct Case {
    pub name: &'static str,
    pub args: &'static [&'static str],
    pub stdin: &'static str,
    pub env: &'static [(&'static str, &'static str)],
    pub files: &'static [(&'static str, &'static str)],
    pub gap: Option<&'static str>,
}

impl Case {
    pub const fn new(name: &'static str, args: &'static [&'static str]) -> Self {
        Self {
            name,
            args,
            stdin: "",
            env: &[],
            files: &[],
            gap: None,
        }
    }

    pub const fn with_stdin(self, stdin: &'static str) -> Self {
        Self { stdin, ..self }
    }

    pub const fn with_env(self, env: &'static [(&'static str, &'static str)]) -> Self {
        Self { env, ..self }
    }

    pub const fn with_files(self, files: &'static [(&'static str, &'static str)]) -> Self {
        Self { files, ..self }
    }

    pub const fn with_gap(self, gap: &'static str) -> Self {
        Self {
            gap: Some(gap),
            ..self
        }
    }
}

pub struct Outcome {
    pub name: &'static str,
    pub gap: Option<&'static str>,
    pub mismatch: Option<String>,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.mismatch.is_none()
    }
}

struct Observed {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: Option<i32>,
}

pub fn run_all(fazo: &Path) -> Result<Vec<Outcome>, String> {
    let lua = reference::shared()?.lua.clone();
    CASES.iter().map(|case| run_case(&lua, fazo, case)).collect()
}

fn run_case(lua: &Path, fazo: &Path, case: &Case) -> Result<Outcome, String> {
    let reference = observe(lua, case)?;
    let candidate = observe(fazo, case)?;
    Ok(Outcome {
        name: case.name,
        gap: case.gap,
        mismatch: difference(&reference, &candidate),
    })
}

fn observe(program: &Path, case: &Case) -> Result<Observed, String> {
    let work = WorkDir::create()?;
    for (name, content) in case.files {
        fs::write(work.path().join(name), content).map_err(|error| format!("cannot write {name}: {error}"))?;
    }
    let captured = reference::run_process(
        program,
        case.args,
        case.stdin.as_bytes(),
        case.env,
        work.path(),
        reference::scaled(CASE_DEADLINE),
    )?;
    Ok(Observed {
        stdout: normalize_stdout(program, &captured.stdout),
        stderr: normalize_stderr(program, &captured.stderr),
        code: captured.code,
    })
}

fn normalize_stdout(program: &Path, bytes: &[u8]) -> Vec<u8> {
    normalize_nan_text(&normalize_names(program, &drop_lines_containing(bytes, BANNER_MARKS)))
}

fn normalize_stderr(program: &Path, bytes: &[u8]) -> Vec<u8> {
    normalize_nan_text(&normalize_names(program, &drop_traceback(bytes)))
}

fn normalize_names(program: &Path, bytes: &[u8]) -> Vec<u8> {
    let path = program.display().to_string();
    let named = replace_all(bytes, path.as_bytes(), NORMALIZED_PROGRAM);
    let placed = match program.parent() {
        Some(directory) => replace_all(&named, directory_text(directory).as_bytes(), NORMALIZED_DIRECTORY),
        None => named,
    };
    replace_all(&placed, PROGRAM_NAME, NORMALIZED_PROGRAM)
}

fn directory_text(directory: &Path) -> String {
    let text = directory.display().to_string();
    if cfg!(windows) { text.replace('/', "\\") } else { text }
}

fn drop_lines_containing(bytes: &[u8], marks: &[&[u8]]) -> Vec<u8> {
    bytes
        .split_inclusive(|&byte| byte == b'\n')
        .filter(|line| marks.iter().all(|mark| find(line, mark).is_none()))
        .flat_map(|line| line.iter().copied())
        .collect()
}

fn drop_traceback(bytes: &[u8]) -> Vec<u8> {
    let mut kept = Vec::with_capacity(bytes.len());
    let mut in_traceback = false;
    for line in bytes.split_inclusive(|&byte| byte == b'\n') {
        if line == TRACEBACK_HEADER {
            in_traceback = true;
            continue;
        }
        if in_traceback && line.starts_with(b"\t") {
            continue;
        }
        in_traceback = false;
        kept.extend_from_slice(line);
    }
    kept
}

fn replace_all(bytes: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut rest = bytes;
    while let Some(at) = find(rest, needle) {
        out.extend_from_slice(&rest[..at]);
        out.extend_from_slice(replacement);
        rest = &rest[at + needle.len()..];
    }
    out.extend_from_slice(rest);
    out
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn difference(reference: &Observed, candidate: &Observed) -> Option<String> {
    if reference.code != candidate.code {
        return Some(format!("exit code: lua {:?}, fazo {:?}", reference.code, candidate.code));
    }
    first_difference("stdout", &reference.stdout, &candidate.stdout)
        .or_else(|| first_difference("stderr", &reference.stderr, &candidate.stderr))
}

fn first_difference(stream: &str, reference: &[u8], candidate: &[u8]) -> Option<String> {
    if reference == candidate {
        return None;
    }
    let mut reference_lines = reference.split_inclusive(|&byte| byte == b'\n');
    let mut candidate_lines = candidate.split_inclusive(|&byte| byte == b'\n');
    let mut number = 1;
    loop {
        match (reference_lines.next(), candidate_lines.next()) {
            (Some(left), Some(right)) if left == right => number += 1,
            (left, right) => {
                return Some(format!(
                    "{stream} line {number}: lua {}, fazo {}",
                    show(left),
                    show(right)
                ));
            }
        }
    }
}

fn show(line: Option<&[u8]>) -> String {
    match line {
        Some(line) => format!("{:?}", String::from_utf8_lossy(line)),
        None => "nothing".to_string(),
    }
}

const CASES: &[Case] = &[
    Case::new("script_prints", &["t.lua"]).with_files(&[("t.lua", "print('script', 1 + 1)\n")]),
    Case::new("script_arguments_are_varargs", &["t.lua", "a", "", "b"])
        .with_files(&[("t.lua", "print(select('#', ...), ...)\n")]),
    Case::new("arg_table_contents", &["t.lua", "x", "", "y"]).with_files(&[(
        "t.lua",
        "print(arg[0], arg[1], arg[2] == '', arg[3], #arg, arg[-1] ~= nil)\n",
    )]),
    Case::new(
        "arg_table_negative_indices",
        &["-e", "print(arg[-3] ~= nil, arg[-2], arg[-1], arg[0], #arg)", "t.lua", "x"],
    )
    .with_files(&[("t.lua", "print(arg[1])\n")]),
    Case::new("script_returns_values_quietly", &["t.lua"]).with_files(&[("t.lua", "return 1, 2\n")]),
    Case::new("script_is_a_directory", &["."]),
    Case::new("missing_script", &["nofile.lua"]),
    Case::new("script_syntax_error", &["t.lua"]).with_files(&[("t.lua", "x = = 1\n")]),
    Case::new("script_runtime_error", &["t.lua"])
        .with_files(&[("t.lua", "print('before')\nerror('boom')\n")]),
    Case::new("script_error_object_table", &["t.lua"]).with_files(&[("t.lua", "error({})\n")]),
    Case::new("script_error_object_with_tostring", &["t.lua"]).with_files(&[(
        "t.lua",
        "error(setmetatable({}, {__tostring = function() return 'custom error' end}))\n",
    )]),
    Case::new("script_error_object_number", &["t.lua"]).with_files(&[("t.lua", "error(42)\n")]),
    Case::new("script_error_object_nil", &["t.lua"]).with_files(&[("t.lua", "error()\n")]),
    Case::new("script_shebang_keeps_line_numbers", &["t.lua"])
        .with_files(&[("t.lua", "#!/usr/bin/env lua\nerror('line two')\n")]),
    Case::new("script_byte_order_mark", &["t.lua"]).with_files(&[("t.lua", "\u{feff}print('bom')\n")]),
    Case::new("stack_overflow_message", &["t.lua"]).with_files(&[("t.lua", OVERFLOW_PROGRAM)]),
    Case::new("stack_overflow_depth", &["t.lua"]).with_files(&[("t.lua", DEPTH_PROGRAM)]),
    Case::new("stack_overflow_depth_with_arguments", &["t.lua", "a", "b"])
        .with_files(&[("t.lua", DEPTH_PROGRAM)]),
    Case::new("stack_overflow_depth_in_e", &["-e", DEPTH_PROGRAM]),
    Case::new("stdin_script", &["-"]).with_stdin("print('from stdin', ...)\n"),
    Case::new("stdin_script_with_arguments", &["-", "p", "q"])
        .with_stdin("print(arg[0], arg[1], arg[2], select('#', ...))\n"),
    Case::new("stdin_script_without_arguments", &[]).with_stdin("print('no arguments')\n"),
    Case::new("stdin_syntax_error", &[]).with_stdin("x = = 1\n"),
    Case::new("stdin_runtime_error", &["-"]).with_stdin("error('stdin boom')\n"),
    Case::new("stdin_dash_after_double_dash_is_a_file", &["--", "-"]),
    Case::new("double_dash_then_script", &["--", "t.lua"]).with_files(&[("t.lua", "print('dd')\n")]),
    Case::new("double_dash_alone_reads_stdin", &["--"]).with_stdin("print('dd stdin')\n"),
    Case::new("e_options_in_order", &["-e", "x = 1", "-e", "print(x)"]),
    Case::new("e_attached_argument", &["-eprint('attached')"]),
    Case::new("e_before_script_shares_globals", &["-e", "y = 5", "t.lua"])
        .with_files(&[("t.lua", "print(y)\n")]),
    Case::new("e_error_stops_the_script", &["-e", "error('e fails')", "t.lua"])
        .with_files(&[("t.lua", "print('should not run')\n")]),
    Case::new("e_then_dash_reads_stdin", &["-e", "x = 5", "-"]).with_stdin("print(x)\n"),
    Case::new("e_alone_does_not_read_stdin", &["-e", "print('only e')"]).with_stdin("print('not read')\n"),
    Case::new("e_syntax_error", &["-e", "x ="]),
    Case::new("version_flag", &["-v"]),
    Case::new("warnings_flag_is_quiet", &["-W", "-e", "print('warned')"]),
    Case::new("warn_with_flag", &["-W", "-e", "warn('careful')"]),
    Case::new("warn_without_flag_is_silent", &["-e", "warn('quiet')"]),
    Case::new("warn_pieces_join_on_one_line", &["-W", "-e", "warn('a', 'b', 1, 2.5)"]),
    Case::new("warn_control_off_then_on", &["-W", "-e", "warn('@off'); warn('hidden'); warn('@on'); warn('shown')"]),
    Case::new("warn_needs_a_string", &["-W", "-e", "warn({})"]),
    Case::new("warn_needs_an_argument", &["-W", "-e", "warn()"]),
    Case::new("library_missing_module", &["-l", "nomodule", "-e", "print(1)"]),
    Case::new("library_named_missing_module", &["-l", "g=nomodule", "-e", "print(1)"]),
    Case::new("library_loads_file_from_working_folder", &["-l", "mylib", "-e", "print(mylib.answer)"])
        .with_files(&[("mylib.lua", "return {answer = 42}\n")]),
    Case::new("usage_unknown_option", &["-x"]),
    Case::new("usage_unknown_long_option", &["--x"]),
    Case::new("usage_e_needs_argument", &["-e"]),
    Case::new("usage_e_followed_by_option", &["-e", "-v"]),
    Case::new("usage_l_needs_argument", &["-l"]),
    Case::new("usage_extra_letters", &["-Ex"]),
    Case::new("lua_init_runs_first", &["t.lua"])
        .with_env(&[("LUA_INIT", "print('init')")])
        .with_files(&[("t.lua", "print('script')\n")]),
    Case::new("lua_init_5_5_wins", &["-e", "print('e')"]).with_env(&[
        ("LUA_INIT_5_5", "print('five')"),
        ("LUA_INIT", "print('four')"),
    ]),
    Case::new("lua_init_ignored_with_e", &["-E", "-e", "print('e')"])
        .with_env(&[("LUA_INIT", "print('init')")]),
    Case::new("lua_init_file", &["-e", "print('after')"])
        .with_env(&[("LUA_INIT", "@init.lua")])
        .with_files(&[("init.lua", "print('from file')\n")]),
    Case::new("lua_init_file_missing", &["-e", "print(1)"]).with_env(&[("LUA_INIT", "@nofile.lua")]),
    Case::new("lua_init_error_name", &["-e", "print(1)"]).with_env(&[("LUA_INIT", "error('init failed')")]),
    Case::new("lua_init_5_5_error_name", &["-e", "print(1)"])
        .with_env(&[("LUA_INIT_5_5", "error('five failed')")]),
    Case::new("lua_init_empty", &["-e", "print('ok')"]).with_env(&[("LUA_INIT", "")]),
    Case::new("lua_init_sees_arg_table", &["t.lua", "a"])
        .with_env(&[("LUA_INIT", "print(arg[0], #arg)")])
        .with_files(&[("t.lua", "print('script')\n")]),
    Case::new("interactive_session", &["-i"]).with_stdin(
        "x = 1\nx\n=x+1\nfunction f()\nreturn 2\nend\nprint(f())\nx = = 2\nerror('boom')\nprint(nil)\n",
    ),
    Case::new("interactive_expressions", &["-i"])
        .with_stdin("1 + 2\n=3 * 4\nlocal t = {1, 2, 3}\n#t\nnil\n"),
    Case::new("interactive_custom_prompts", &["-i"])
        .with_stdin("_PROMPT = 'P> '\n_PROMPT2 = 'C> '\nx = 1 +\n2\nprint(x)\n"),
    Case::new("interactive_multiline_function", &["-i"])
        .with_stdin("function g(a)\nreturn a * 2\nend\nprint(g(21))\n"),
    Case::new("interactive_eof_inside_block", &["-i"]).with_stdin("if true then\n"),
    Case::new("interactive_unfinished_string", &["-i"]).with_stdin("x = 'abc\nprint('next')\n"),
    Case::new("interactive_print_failure", &["-i"])
        .with_stdin("setmetatable({}, {__tostring = function() error('bad') end})\n"),
    Case::new("interactive_empty_lines", &["-i"]).with_stdin("\n\n"),
    Case::new("interactive_last_line_without_newline", &["-i"]).with_stdin("print(7)"),
    Case::new("interactive_stack_overflow_depth", &["-i"]).with_stdin(
        "n = 0\nfunction f() n = n + 1; io.write(n, \"\\n\"); return 1 + f() end\nf()\n",
    ),
    Case::new("exit_close_main_variables", &["t.lua"]).with_files(&[(
        "t.lua",
        r##"local sentinel = setmetatable({}, {__gc = function() print("gc") end})
local main_close <close> = setmetatable({}, {__close = function(_, err)
  print("closed main", err)
end})
local function inner()
  local nested <close> = setmetatable({}, {__close = function(_, err)
    print("closed nested", err)
  end})
  os.exit(3, true)
end
print("before exit")
inner()
"##,
    )]),
    Case::new("exit_close_error_chain", &["t.lua"]).with_files(&[(
        "t.lua",
        r##"local first <close> = setmetatable({}, {__close = function(_, err)
  print("first", select("#", err), err)
end})
local failing <close> = setmetatable({}, {__close = function(_, err)
  print("failing", select("#", err), err)
  error("close failed", 0)
end})
local last <close> = setmetatable({}, {__close = function(...)
  print("last", select("#", ...))
  error(setmetatable({}, {__tostring = function() return "object" end}))
end})
local function swallow()
  local ignored <close> = setmetatable({}, {__close = function(_, err)
    print("ignored", err)
  end})
  return "unreachable"
end
print(pcall(swallow))
os.exit(0, true)
"##,
    )]),
    Case::new("exit_close_suspended_coroutine", &["t.lua"]).with_files(&[(
        "t.lua",
        r##"local co = coroutine.create(function()
  local pending <close> = setmetatable({}, {__close = function() print("coroutine closed") end})
  coroutine.yield("paused")
  print("resumed")
end)
print(coroutine.resume(co))
local outer <close> = setmetatable({}, {__close = function() print("main closed") end})
os.exit(true, true)
"##,
    )]),
];

#[cfg(test)]
mod tests {
    use super::{first_difference, drop_traceback, normalize_names};
    use std::path::Path;

    #[test]
    fn traceback_blocks_are_removed_and_messages_kept() {
        let text = b"lua: t.lua:1: boom\nstack traceback:\n\t[C]: in function 'error'\n\tt.lua:1: in main chunk\n\t[C]: in ?\n";
        assert_eq!(drop_traceback(text), b"lua: t.lua:1: boom\n");
    }

    #[test]
    fn program_names_are_normalized() {
        let program = Path::new("/bin/lua");
        assert_eq!(
            normalize_names(program, b"/bin/lua: x\nfazo: y\n"),
            b"<prog>: x\n<prog>: y\n"
        );
    }

    #[test]
    fn the_first_differing_line_is_reported() {
        assert_eq!(first_difference("stdout", b"a\nb\n", b"a\nb\n"), None);
        assert_eq!(
            first_difference("stdout", b"a\nb\n", b"a\nc\n"),
            Some("stdout line 2: lua \"b\\n\", fazo \"c\\n\"".to_string())
        );
        assert_eq!(
            first_difference("stdout", b"a\n", b"a\nextra\n"),
            Some("stdout line 2: lua nothing, fazo \"extra\\n\"".to_string())
        );
    }
}
