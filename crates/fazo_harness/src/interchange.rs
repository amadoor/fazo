use std::fmt::Write as _;
use std::{env, fs, path::PathBuf, time::Duration};

use crate::{
    differential::{run_fazo_with_standard_libraries, run_reference},
    reference::{self, WorkDir},
};

const SCRIPT: &str = "prog.lua";
const CHUNK: &str = "prog.luac";
const DRIVER: &str = "driver.lua";
const DEADLINE: Duration = Duration::from_secs(120);

const PROGRAMS: &[(&str, &str)] = &[
    (
        "numbers_and_strings",
        "print(1, 2.5, -0.0, 2^53, 10 // 3, 7 % -3, 'a' .. 'b', #'xyz', math.maxinteger, 1e300 * 1e10)\n",
    ),
    (
        "closures_and_loops",
        "local fs = {}\n\
         for i = 1, 3 do local j = i * 2; fs[i] = function() j = j + 1; return i, j end end\n\
         for _, f in ipairs(fs) do print(f()) end\n\
         local k = 0\n\
         while true do k = k + 1; if k > 5 then break end end\n\
         print(k)\n\
         for i = 1, 3 do\n\
           for j = 1, 3 do\n\
             if j == 2 then goto continue end\n\
             print(i, j)\n\
             ::continue::\n\
           end\n\
         end\n",
    ),
    (
        "constants_and_tables",
        "local K <const> = 10\n\
         local S <const> = 'key'\n\
         local F <const> = 2.0\n\
         local t = {S = K, [K] = 1.5, 1, 2, 3, F}\n\
         print(t.key, t[10], t[1], #t, K + 1.0, F * 0.5, t[F])\n\
         local big = {}\n\
         for i = 1, 300 do big[#big + 1] = i end\n\
         print(#big, big[300], select('#', table.unpack(big, 1, 250)))\n",
    ),
    (
        "varargs_and_to_be_closed",
        "local function f(...) local a, b = ... return select('#', ...), a, b end\n\
         print(f(1, nil, 3))\n\
         do\n\
           local x <close> = setmetatable({}, {__close = function() print('closed') end})\n\
           print('body')\n\
         end\n\
         local function g(...) return ... end\n\
         print(g(4, 5, 6))\n",
    ),
    (
        "globals_and_long_lines",
        "global *\n\
         global g1, g2 = 1, 2\n\
         global function h() return g1 + g2 end\n\
         print(h())\n\
         local t = 0\n\
         t = t + 1\n\
         t = t + 2\n\
         t = t + 3\n\
         t = t + 4\n\
         t = t + 5\n\
         t = t + 6\n\
         t = t + 7\n\
         t = t + 8\n\
         t = t + 9\n\
         t = t + 10\n\
         t = t + 11\n\
         t = t + 12\n\
         t = t + 13\n\
         t = t + 14\n\
         t = t + 15\n\
         t = t + 16\n\
         t = t + 17\n\
         t = t + 18\n\
         t = t + 19\n\
         t = t + 20\n\
         t = t + 21\n\
         t = t + 22\n\
         t = t + 23\n\
         t = t + 24\n\
         t = t + 25\n\
         t = t + 26\n\
         t = t + 27\n\
         t = t + 28\n\
         t = t + 29\n\
         t = t + 30\n\
         t = t + 31\n\
         t = t + 32\n\
         t = t + 33\n\
         t = t + 34\n\
         t = t + 35\n\
         t = t + 36\n\
         t = t + 37\n\
         t = t + 38\n\
         t = t + 39\n\
         t = t + 40\n\
         t = t + 41\n\
         t = t + 42\n\
         t = t + 43\n\
         t = t + 44\n\
         t = t + 45\n\
         t = t + 46\n\
         t = t + 47\n\
         t = t + 48\n\
         t = t + 49\n\
         t = t + 50\n\
         t = t + 51\n\
         t = t + 52\n\
         t = t + 53\n\
         t = t + 54\n\
         t = t + 55\n\
         t = t + 56\n\
         t = t + 57\n\
         t = t + 58\n\
         t = t + 59\n\
         t = t + 60\n\
         t = t + 61\n\
         t = t + 62\n\
         t = t + 63\n\
         t = t + 64\n\
         t = t + 65\n\
         t = t + 66\n\
         t = t + 67\n\
         t = t + 68\n\
         t = t + 69\n\
         t = t + 70\n\
         print(t)\n",
    ),
];

fn reference_chunk(source: &[u8], strip: bool) -> Result<Vec<u8>, String> {
    let lua = reference::shared()?;
    let work = WorkDir::create()?;
    fs::write(work.path().join(SCRIPT), source).map_err(|error| format!("cannot write the script: {error}"))?;
    let mut args = Vec::new();
    if strip {
        args.push("-s");
    }
    args.extend(["-o", CHUNK, SCRIPT]);
    let captured = reference::execute(&lua.luac, &args, work.path(), DEADLINE)?;
    if !captured.success {
        return Err(String::from_utf8_lossy(&captured.stderr).into_owned());
    }
    fs::read(work.path().join(CHUNK)).map_err(|error| format!("cannot read the chunk: {error}"))
}

fn lua_bytes(bytes: &[u8]) -> String {
    let mut literal = String::from("\"");
    for byte in bytes {
        let _ = write!(literal, "\\{byte:03}");
    }
    literal.push('"');
    literal
}

#[test]
fn reference_chunks_run_in_fazo() {
    for (name, source) in PROGRAMS {
        let expected = run_reference(source.as_bytes(), SCRIPT).expect("the reference runs the program");
        assert!(expected.error.is_none(), "{name}: the reference fails: {}", expected.render());
        for strip in [false, true] {
            let chunk = reference_chunk(source.as_bytes(), strip).expect("the reference writes the chunk");
            let driver = format!("local f = assert(load({}, '=prog', 'b'))\nf()\n", lua_bytes(&chunk));
            let outcome = run_fazo_with_standard_libraries(driver.as_bytes(), "driver");
            assert_eq!(outcome.error, None, "{name}, strip {strip}");
            assert_eq!(outcome.output, expected.output, "{name}, strip {strip}");
        }
    }
}

/// The in-process VM has no file system, so the executable writes the chunk.
fn fazo_executable() -> Result<PathBuf, String> {
    env::var_os("FAZO_CLI")
        .map(PathBuf::from)
        .ok_or_else(|| "set FAZO_CLI to the fazo executable (crates/fazo_cli)".to_string())
}

#[test]
fn fazo_chunks_run_in_reference() {
    let fazo = fazo_executable().expect("the fazo executable");
    let lua = reference::shared().expect("the reference interpreter");
    for (name, source) in PROGRAMS {
        let expected = run_reference(source.as_bytes(), SCRIPT).expect("the reference runs the program");
        assert!(expected.error.is_none(), "{name}: the reference fails: {}", expected.render());
        for strip in [false, true] {
            let work = WorkDir::create().expect("a work directory");
            let written = work.path().join(CHUNK).display().to_string();
            let driver = format!(
                "local out = assert(io.open({written:?}, 'wb'))\n\
                 out:write(string.dump(assert(load([==[{source}]==], '=prog')), {strip}))\n\
                 out:close()\n"
            );
            fs::write(work.path().join(DRIVER), driver).expect("the driver is written");
            let driven = reference::execute(&fazo, &[DRIVER], work.path(), DEADLINE).expect("fazo runs the driver");
            assert!(driven.success, "{name}, strip {strip}: writing the chunk: {}", String::from_utf8_lossy(&driven.stderr));
            let captured = reference::execute(&lua.lua, &[CHUNK], work.path(), DEADLINE).expect("the reference runs");
            assert!(captured.success, "{name}, strip {strip}: the reference rejects the chunk: {}", String::from_utf8_lossy(&captured.stderr));
            assert_eq!(captured.stdout, expected.output, "{name}, strip {strip}");
        }
    }
}
