# fazo_harness

The differential harness for `fazo`: it runs the official PUC-Rio Lua 5.5.1 interpreter and fazo on the
same programs and reports where they differ. It writes `crates/fazo/scoreboard.json`.

Nothing from lua.org is checked into the repository. The harness fetches the sources and the official
test suite at run time, verifies them against pinned SHA-256 hashes, and builds the standalone `lua`
executable from the sources.

## Pinned versions

| Archive | URL | SHA-256 |
|---|---|---|
| Lua 5.5.1 sources | `https://www.lua.org/ftp/lua-5.5.1.tar.gz` | `1c4b4068d67061f2a2231ad2b5422e77acea1487ea9890f6320af614f4373dce` |
| Lua 5.5.1 test suite | `https://www.lua.org/tests/lua-5.5.1-tests.tar.gz` | `da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31` |

The hashes are pinned in `src/reference.rs`. A download whose hash differs is a hard error and nothing
is extracted.

`cargo xtask perf` also measures LuaJIT 2.1 (MIT), pinned by commit in `src/perf.rs`:
`a4c064539ea931274d407e1249b4bb575424a0f4` from `https://github.com/LuaJIT/LuaJIT.git`. Nothing from
LuaJIT is checked into the repository.

## Cache location

The cache is `fazo-reference` inside the cargo target directory: `$CARGO_TARGET_DIR/fazo-reference`, or
`target/fazo-reference` at the workspace root when `CARGO_TARGET_DIR` is unset. `FAZO_REFERENCE_DIR`
overrides the whole folder. It holds:

- `lua-5.5.1/` and `lua-5.5.1-tests/`, the extracted archives;
- `bin/lua` (`lua.exe` on Windows), the built standalone interpreter;
- `work/`, the folders each run writes its script into, removed after the run.

## Populating the cache

The harness fills the cache when it is missing. To fetch the sources without building or running
anything:

```
cargo run --manifest-path crates/fazo_harness/Cargo.toml --bin scoreboard -- fetch
```

To run the whole scoreboard, which builds the interpreter if it is missing:

```
cargo run --manifest-path crates/fazo_harness/Cargo.toml --bin scoreboard -- --seeds 200
```

The unit tests that run the reference need the cache too. They fetch and build it on first use, so a
first `cargo test` needs network access. Once the cache exists they work offline.

## The scoreboard file

`crates/fazo/scoreboard.json` holds the latest results, pretty-printed with keys in a fixed order:

- `lua`: the PUC-Rio Lua version the results compare against.
- `tests.<suite>.<run>`: `pass`, `fail` and `skip` counts and the names of the failing files or cases, for the
  suites `official`, `cases`, `syntax`, `cli` and `generated`. The run is the `--jit` tier (`normal`,
  `interpreter`, `baseline`, `optimizing`, `peak`), or the name given with `--entry` (`gc-stress`, `debug`). A
  generated failure is named `<profile>/<seed>`. A case or seed that runs past its deadline is a failure
  named `<case> (timeout)`. `skip` counts out-of-scope official files, command-line
  cases with a known gap, and generated programs PUC Lua rejects.
- `bytecode`: written by `cargo xtask bytecode`.
- `perf` and `operations`: written by `cargo xtask perf`, in Valgrind instructions net of the empty script.
  `perf.<workload>` holds `puc`, `fazo` (normal tiering), `baseline` (`-X tier=baseline`), `optimizing`
  (`-X force=optimizing`), `peak` (`-X force=peak`), `interpreter` (`-X tier=interpreter`), `luajit` (JIT
  on) and `luajit_interpreter` (`-joff`), and `split`: normal tiering's Ir inside compiles (`compile`),
  before the first Optimizing or Peak code runs (`before_optimized`), and the rest (`steady`). A workload
  LuaJIT cannot run has `null` LuaJIT numbers and the reason in `luajit_missing`. `operations.<op>` holds
  the per-statement count for `puc`, `fazo` (the interpreter), `baseline`, `optimizing`, `peak` and
  `luajit`, and the interpreter's budget from `perf/ops/budgets.txt`. `compile_cost` holds the Ir per
  bytecode of a 500-statement function compiled by Peak, by Optimizing and by LuaJIT.

Each run reads the file, replaces only its own entries and writes it back.

## Investigating generated mismatches

`scoreboard mismatches full|core --seeds N` prints one line per mismatching seed with the first
differing output line or fazo's error. `scoreboard minimize full|core SEED` prints the program for
one seed reduced to the lines the mismatch needs. Both run the official interpreter, so they need
the cache.

## How the interpreter is built

The standalone `lua` target of the official makefile, reproduced with the `cc` crate's compiler
driver: every library source of `src/` plus `lua.c` (`luac.c` is not part of `lua`), compiled with
`-std=gnu99 -O2 -Wall -Wextra -DLUA_COMPAT_5_3`, the platform define the default `make` target uses
(`LUA_USE_LINUX` on Linux, `LUA_USE_MACOSX` on macOS, none on Windows), and linked with `-lm`
(plus `-ldl` and `-Wl,-E` on Linux, and `-lreadline` on macOS). MSVC gets the same defines through `cl`. The target and host are
the running platform, and the optimization level is 2.

## How the reference runs

Scripts run as `lua <name>` in a fresh folder, so messages carry the script's name. Official files run
as `lua -e "_port=true;_soft=true" <name>` in the suite folder. Stdout is compared exactly, with CRLF
turned into LF on Windows. For errors the harness compares the first stderr line without the program
name the interpreter prints before it; the traceback is not compared.

Every run has a deadline that `--timeout-scale N` multiplies: 60 seconds for a hand-written case, syntax
case or generated seed, 120 seconds for an official file or command-line case. Processes are killed at
their deadline. fazo runs the cases, syntax cases and seeds in-process, and a timer thread stops them
through the VM's interrupt handle.

## Performance

`cargo xtask perf` runs Valgrind's cachegrind through WSL on Windows and directly on Linux. It builds
three interpreters in `FAZO_PERF_BUILD_DIR` (default `/tmp/fazo-perf`; a Windows path is turned into its
`/mnt/<drive>/...` form):

- PUC Lua 5.5.1 from the reference sources, as `bin/lua`;
- fazo with the `profiling` profile, as `target/profiling/fazo`;
- LuaJIT with `make amalg`, as `luajit-<commit>/src/luajit`. The sources come from `git archive` of the
  pinned commit, taken from the checkout `FAZO_LUAJIT_DIR` names, or fetched into `luajit-git/` when it
  is unset. LuaJIT is built without `-DLUAJIT_USE_VALGRIND`: that define discards Valgrind's translations
  after each machine-code write (`src/lj_mcode.c`) and quiets memcheck reports, and Valgrind's default
  `--smc-check=all-non-file` already retranslates code written into anonymous mappings.

Each workload in `perf/` runs on every interpreter, and the run fails when an output differs from PUC
Lua's. LuaJIT implements Lua 5.1, so a workload that needs Lua 5.5 has a `<name>.luajit.lua` beside it
that does the same work in Lua 5.1: floats where the integers stay exact, and `bit.*` for bitwise
operations. A workload with no such version reports why it has no LuaJIT number. The console prints
fazo's ratio to PUC Lua and to LuaJIT, then lists the workloads where fazo is slower than LuaJIT; only
the PUC ratio and the operation budgets fail the run.

## Quick check

`cargo xtask quick [seeds]` is the check to run after every compiler change; it takes a few minutes and
needs neither Valgrind nor the reference cache. It runs:

- `cargo test -p fazo --lib`;
- every workload in `perf/` under `-X force=peak`, `-X force=optimizing`, each of those with
  `stress-exit=3 stress-ic`, and under default tiering, each compared with its output under
  `-X tier=interpreter`;
- every per-operation script in `perf/ops/` under `-X force=peak` and `-X force=optimizing`, compared with
  the interpreter;
- a deopt and recompile report from the JIT's own counters (`-X jitstats`), which counts as a failure any
  deopt site that fired 8 times or more, any prototype compiled by Optimizing and Peak 6 times or more, and any prototype
  recompiled 8 times or more;
- `seeds` generated programs per generator profile (Full, Metatables and Core; default 100) under
  `-X force=peak` and `-X force=optimizing`, compared with the interpreter;
- the official `math`, `calls`, `nextvar`, `coroutine` and `sort` tests under `-X force=peak` and
  `-X force=optimizing`.

It builds fazo with the `profiling` profile and prints each step's time and every failure.

## Command-line cases

`src/cli.rs` holds the cases for the `fazo` executable (`crates/fazo_cli`). Each case runs the official
`lua` and `fazo` as processes with the same arguments, standard input, environment and files, in fresh
folders. Stdout and the exit code must match exactly. Stderr must match exactly except traceback blocks,
which fazo does not print until F5. The program path and the program name are normalized to `<prog>`,
and the version banner is left out of the comparison. A case with a `gap` reason is reported but does
not fail the tests until its feature exists.

Build the executable before the scoreboard runs, or point `FAZO_CLI` at it:

```
cargo build --manifest-path crates/fazo_cli/Cargo.toml
```

The `fazo_cli` tests run every case, and the scoreboard records them under `tests.cli`, together with the
standalone `main.lua` run.

## Scope

`api.lua` (needs the C test library `T`) and `heavy.lua` (too long for the deadline) are out of scope for
fazo. `heavy.lua` is not run on the reference. `main.lua` passes in portable mode, where it returns at
its first line; the standalone run that drives the executable through `os` and `io` is recorded under
`tests.cli` as `standalone main.lua`.
