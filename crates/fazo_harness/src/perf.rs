use std::fmt::Write as _;
use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    process::{Command, Output},
};

use crate::{reference, report};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const DEFAULT_BUILD_DIR: &str = "/tmp/fazo-perf";
const EMPTY_SCRIPT: &str = "empty.lua";
const INTERPRETER_ONLY: &[&str] = &["-X", "tier=interpreter"];
const BASELINE_ONLY: &[&str] = &["-X", "tier=baseline"];
const PEAK_ONLY: &[&str] = &["-X", "force=peak"];
const OPERATIONS_DIR: &str = "ops";
const BUDGETS_FILE: &str = "budgets.txt";
const OPERATIONS_PER_RUN: f64 = 2_000_000.0;
const OPERATION_TOLERANCE: f64 = 1.0;
const LUAJIT_COMMIT: &str = "a4c064539ea931274d407e1249b4bb575424a0f4";
const LUAJIT_REPOSITORY: &str = "https://github.com/LuaJIT/LuaJIT.git";
const LUAJIT_DIR_VARIABLE: &str = "FAZO_LUAJIT_DIR";
const LUAJIT_JIT_OFF: &[&str] = &["-joff"];
const LUAJIT_SUFFIX: &str = ".luajit.lua";
const NO_GENERATIONAL_COLLECTOR: &str =
    "LuaJIT 2.1 has no generational collector: collectgarbage(\"generational\") is an invalid option";

/// How LuaJIT runs a workload: the script itself, its Lua 5.1 version `<name>.luajit.lua`, or not at all.
pub enum Luajit {
    Unchanged,
    Ported,
    Missing(&'static str),
}

pub struct Workload {
    pub name: &'static str,
    pub script: &'static str,
    pub arguments: &'static [&'static str],
    pub luajit: Luajit,
}

impl Workload {
    pub fn luajit_script(&self) -> Option<String> {
        match self.luajit {
            Luajit::Unchanged => Some(self.script.to_owned()),
            Luajit::Ported => Some(format!("{}{LUAJIT_SUFFIX}", self.script.trim_end_matches(".lua"))),
            Luajit::Missing(_) => None,
        }
    }

    fn luajit_missing(&self) -> Option<&'static str> {
        match self.luajit {
            Luajit::Missing(reason) => Some(reason),
            Luajit::Unchanged | Luajit::Ported => None,
        }
    }
}

pub const WORKLOADS: &[Workload] = &[
    Workload { name: "fib", script: "fib.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "integer loop", script: "int_loop.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "float loop", script: "float_loop.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "table array", script: "table_array.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "table hash", script: "table_hash.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "closures", script: "closures.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "method calls through __index", script: "method_index.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "absent metamethods", script: "absent_metamethods.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "coroutine switching", script: "coroutine_switch.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "strings", script: "strings.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "binary-trees", script: "binary_trees.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "n-body", script: "run_awfy.lua", arguments: &["nbody", "12000"], luajit: Luajit::Unchanged },
    Workload { name: "spectral-norm", script: "spectral_norm.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "fannkuch", script: "fannkuch.lua", arguments: &[], luajit: Luajit::Unchanged },
    Workload { name: "class dispatch", script: "class_dispatch.lua", arguments: &[], luajit: Luajit::Ported },
    Workload { name: "deltablue", script: "run_awfy.lua", arguments: &["deltablue", "1000"], luajit: Luajit::Unchanged },
    Workload { name: "havlak", script: "run_awfy.lua", arguments: &["havlak_sized", "1"], luajit: Luajit::Unchanged },
    Workload { name: "garbage collector (generational)", script: "gc_heap.lua", arguments: &[], luajit: Luajit::Missing(NO_GENERATIONAL_COLLECTOR) },
    Workload { name: "stats", script: "stats.lua", arguments: &[], luajit: Luajit::Ported },
    Workload { name: "dialogue (interpreted)", script: "dialogue.lua", arguments: &["interpreted"], luajit: Luajit::Ported },
    Workload { name: "dialogue (compiled with load)", script: "dialogue.lua", arguments: &["compiled"], luajit: Luajit::Ported },
    Workload { name: "frame", script: "frame.lua", arguments: &[], luajit: Luajit::Ported },
];

struct Budget {
    name: String,
    budget_ir: f64,
    recorded_ir: f64,
    reason: String,
}

#[derive(Clone, Copy)]
struct OperationCounts {
    puc: f64,
    interpreter: f64,
    baseline: f64,
    peak: f64,
    luajit: f64,
}

struct Operation<'a> {
    budget: &'a Budget,
    counts: OperationCounts,
}

#[derive(Clone, Copy)]
struct LuajitCounts {
    jit: f64,
    interpreter: f64,
}

#[derive(Clone, Copy)]
struct Counts {
    puc: f64,
    fazo: f64,
    baseline: f64,
    peak: f64,
    interpreter: f64,
    luajit: Option<LuajitCounts>,
}

struct Measurement {
    workload: &'static Workload,
    counts: Counts,
}

struct Interpreters {
    puc: String,
    fazo: String,
    luajit: String,
}

struct Scripts<'a> {
    fazo: &'a Path,
    luajit: Option<&'a Path>,
    arguments: &'a [&'a str],
}

pub fn run() -> Result<(), String> {
    reference::ensure()?;
    let out_dir = reference::cache_dir().join("perf");
    fs::create_dir_all(&out_dir).map_err(|error| format!("creating {}: {error}", out_dir.display()))?;
    let interpreters = build_interpreters(&reference::cache_dir().join(reference::SOURCE_DIR))?;
    let scripts = Path::new(CRATE_DIR).join("perf");
    let operations_dir = scripts.join(OPERATIONS_DIR);
    let budgets = read_budgets(&operations_dir)?;
    check_every_statement_budgeted(&operations_dir, &budgets)?;

    let measurements = measure_workloads(&interpreters, &out_dir, &scripts)?;
    let operations = measure_operations(&interpreters, &out_dir, &operations_dir, &budgets)?;
    let path = report::update(|board| record(board, &measurements, &operations))?;
    print!("{}", summary(&measurements, &operations));
    println!("report: {}", path.display());
    print!("{}", compare_with_luajit(&measurements));

    let failures: Vec<String> = [check_workloads(&measurements), check_operations(&operations)]
        .into_iter()
        .filter_map(Result::err)
        .collect();
    if failures.is_empty() {
        return Ok(());
    }
    Err(failures.join("\n"))
}

fn measure_workloads(interpreters: &Interpreters, out_dir: &Path, scripts: &Path) -> Result<Vec<Measurement>, String> {
    let empty_script = scripts.join(EMPTY_SCRIPT);
    let empty_inputs = Scripts {
        fazo: &empty_script,
        luajit: Some(&empty_script),
        arguments: &[],
    };
    let empty = Counts::measure(interpreters, out_dir, "empty", &empty_inputs)?;

    let mut measurements = Vec::with_capacity(WORKLOADS.len());
    for workload in WORKLOADS {
        let script = scripts.join(workload.script);
        let luajit_script = workload.luajit_script().map(|name| scripts.join(name));
        let inputs = Scripts {
            fazo: &script,
            luajit: luajit_script.as_deref(),
            arguments: workload.arguments,
        };
        check_same_output(interpreters, workload.name, &inputs)?;
        let counts = Counts::measure(interpreters, out_dir, &slug(workload.name), &inputs)?;
        measurements.push(Measurement {
            workload,
            counts: counts.minus(&empty),
        });
    }
    Ok(measurements)
}

impl Counts {
    fn measure(interpreters: &Interpreters, out_dir: &Path, label: &str, scripts: &Scripts) -> Result<Self, String> {
        let run = |program: &str, options: &[&str], kind: &str, script: &Path| {
            measure(program, options, out_dir, &format!("{kind}-{label}"), script, scripts.arguments)
        };
        let script = scripts.fazo;
        let luajit = scripts
            .luajit
            .map(|script| -> Result<LuajitCounts, String> {
                Ok(LuajitCounts {
                    jit: run(&interpreters.luajit, &[], "luajit", script)?,
                    interpreter: run(&interpreters.luajit, LUAJIT_JIT_OFF, "luajit-interpreter", script)?,
                })
            })
            .transpose()?;

        Ok(Self {
            puc: run(&interpreters.puc, &[], "puc", script)?,
            fazo: run(&interpreters.fazo, &[], "fazo", script)?,
            baseline: run(&interpreters.fazo, BASELINE_ONLY, "baseline", script)?,
            peak: run(&interpreters.fazo, PEAK_ONLY, "peak", script)?,
            interpreter: run(&interpreters.fazo, INTERPRETER_ONLY, "interpreter", script)?,
            luajit,
        })
    }

    fn minus(&self, empty: &Self) -> Self {
        let luajit = self.luajit.zip(empty.luajit).map(|(counts, empty)| LuajitCounts {
            jit: counts.jit - empty.jit,
            interpreter: counts.interpreter - empty.interpreter,
        });
        Self {
            puc: self.puc - empty.puc,
            fazo: self.fazo - empty.fazo,
            baseline: self.baseline - empty.baseline,
            peak: self.peak - empty.peak,
            interpreter: self.interpreter - empty.interpreter,
            luajit,
        }
    }
}

fn record(board: &mut report::Scoreboard, measurements: &[Measurement], operations: &[Operation]) {
    board.perf = measurements
        .iter()
        .map(|measurement| (measurement.workload.name.to_owned(), workload_entry(measurement)))
        .collect();
    board.operations = operations
        .iter()
        .map(|operation| (operation.budget.name.clone(), operation_entry(operation)))
        .collect();
}

fn workload_entry(measurement: &Measurement) -> report::Workload {
    let counts = &measurement.counts;
    report::Workload {
        puc: whole(counts.puc),
        fazo: whole(counts.fazo),
        baseline: whole(counts.baseline),
        peak: whole(counts.peak),
        interpreter: whole(counts.interpreter),
        luajit: counts.luajit.map(|luajit| whole(luajit.jit)),
        luajit_interpreter: counts.luajit.map(|luajit| whole(luajit.interpreter)),
        luajit_missing: measurement.workload.luajit_missing().map(str::to_owned),
    }
}

fn operation_entry(operation: &Operation) -> report::Operation {
    let counts = &operation.counts;
    report::Operation {
        puc: tenths(counts.puc),
        fazo: tenths(counts.interpreter),
        baseline: tenths(counts.baseline),
        peak: tenths(counts.peak),
        luajit: tenths(counts.luajit),
        budget: tenths(operation.budget.budget_ir),
    }
}

fn whole(value: f64) -> i64 {
    value.round() as i64
}

fn tenths(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn summary(measurements: &[Measurement], operations: &[Operation]) -> String {
    let mut text = String::new();
    for measurement in measurements {
        let _ = writeln!(text, "{}", workload_line(measurement));
    }
    for operation in operations {
        let counts = &operation.counts;
        let _ = writeln!(
            text,
            "{}: PUC {:.1}, interpreter {:.1}, baseline {:.1}, peak {:.1}, LuaJIT {:.1}, budget {:.1}",
            operation.budget.name, counts.puc, counts.interpreter, counts.baseline, counts.peak, counts.luajit, operation.budget.budget_ir,
        );
    }
    text
}

fn workload_line(measurement: &Measurement) -> String {
    let counts = &measurement.counts;
    let luajit = match (counts.luajit, measurement.workload.luajit_missing()) {
        (Some(luajit), _) => format!(
            "LuaJIT {:.0}, LuaJIT -joff {:.0}, fazo/LuaJIT {:.3}",
            luajit.jit,
            luajit.interpreter,
            counts.fazo / luajit.jit
        ),
        (None, reason) => format!("no LuaJIT number ({})", reason.unwrap_or_default()),
    };
    format!(
        "{}: PUC {:.0}, fazo {:.0}, baseline {:.0}, peak {:.0}, interpreter {:.0}, fazo/PUC {:.3}, {luajit}",
        measurement.workload.name,
        counts.puc,
        counts.fazo,
        counts.baseline,
        counts.peak,
        counts.interpreter,
        measurement.instruction_ratio(),
    )
}

impl Measurement {
    fn instruction_ratio(&self) -> f64 {
        self.counts.fazo / self.counts.puc
    }

    fn luajit_ratio(&self) -> Option<f64> {
        self.counts.luajit.map(|luajit| self.counts.fazo / luajit.jit)
    }
}

fn check_workloads(measurements: &[Measurement]) -> Result<(), String> {
    let failing: Vec<&str> = measurements
        .iter()
        .filter(|measurement| measurement.instruction_ratio() > 1.0)
        .map(|measurement| measurement.workload.name)
        .collect();
    if failing.is_empty() {
        println!("workloads: every workload at or below PUC Lua");
        return Ok(());
    }
    Err(format!("workloads: above PUC Lua: {}", failing.join(", ")))
}

fn compare_with_luajit(measurements: &[Measurement]) -> String {
    let slower: Vec<String> = measurements
        .iter()
        .filter_map(|measurement| Some((measurement.workload.name, measurement.luajit_ratio()?)))
        .filter(|&(_, ratio)| ratio > 1.0)
        .map(|(name, ratio)| format!("{name} ({ratio:.3}x)"))
        .collect();
    let missing: Vec<String> = measurements
        .iter()
        .filter_map(|measurement| Some(format!("{} ({})", measurement.workload.name, measurement.workload.luajit_missing()?)))
        .collect();

    let mut text = String::new();
    if slower.is_empty() {
        text.push_str("LuaJIT: fazo's JIT at or below LuaJIT on every workload LuaJIT runs\n");
    } else {
        let _ = writeln!(text, "LuaJIT: fazo's JIT slower than LuaJIT on {}", slower.join(", "));
    }
    if !missing.is_empty() {
        let _ = writeln!(text, "LuaJIT: no number for {}", missing.join(", "));
    }
    text
}

impl Operation<'_> {
    fn over_budget(&self) -> bool {
        self.budget.reason.is_empty() && self.counts.interpreter > self.budget.budget_ir + OPERATION_TOLERANCE
    }

    fn over_recorded(&self) -> bool {
        self.counts.interpreter > self.budget.recorded_ir + OPERATION_TOLERANCE
    }

    fn describe(&self) -> String {
        format!(
            "{} (fazo {:.1}, budget {:.1}, recorded {:.1})",
            self.budget.name, self.counts.interpreter, self.budget.budget_ir, self.budget.recorded_ir
        )
    }
}

fn check_operations(operations: &[Operation]) -> Result<(), String> {
    let failing: Vec<String> = operations
        .iter()
        .filter(|operation| operation.over_budget() || operation.over_recorded())
        .map(Operation::describe)
        .collect();
    if failing.is_empty() {
        println!("operations: every operation within its budget");
        return Ok(());
    }
    Err(format!("operations: above budget or recorded: {}", failing.join(", ")))
}

fn read_budgets(dir: &Path) -> Result<Vec<Budget>, String> {
    let path = dir.join(BUDGETS_FILE);
    let text = fs::read_to_string(&path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    text.lines().filter(|line| !line.trim().is_empty()).map(parse_budget).collect()
}

fn parse_budget(line: &str) -> Result<Budget, String> {
    let mut fields = line.splitn(4, ' ');
    let name = fields.next();
    let budget_ir = fields.next().and_then(|field| field.parse().ok());
    let recorded_ir = fields.next().and_then(|field| field.parse().ok());
    let reason = fields.next().unwrap_or_default();
    match (name, budget_ir, recorded_ir) {
        (Some(name), Some(budget_ir), Some(recorded_ir)) => Ok(Budget {
            name: name.to_owned(),
            budget_ir,
            recorded_ir,
            reason: reason.to_owned(),
        }),
        _ => Err(format!("{BUDGETS_FILE} line is not `name budget recorded reason`: {line}")),
    }
}

fn check_every_statement_budgeted(dir: &Path, budgets: &[Budget]) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|error| format!("reading {}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| format!("reading {}: {error}", dir.display()))?.path();
        let file = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        let Some(name) = file.strip_suffix(".lua") else {
            continue;
        };
        if file == EMPTY_SCRIPT || budgets.iter().any(|budget| budget.name == name) {
            continue;
        }
        return Err(format!("{file} has no line in {BUDGETS_FILE}"));
    }
    Ok(())
}

fn measure_operations<'a>(
    interpreters: &Interpreters,
    out_dir: &Path,
    dir: &Path,
    budgets: &'a [Budget],
) -> Result<Vec<Operation<'a>>, String> {
    let empty = OperationCounts::measure(interpreters, out_dir, "op-empty", &dir.join(EMPTY_SCRIPT))?;

    let mut operations = Vec::with_capacity(budgets.len());
    for budget in budgets {
        let script = dir.join(format!("{}.lua", budget.name));
        let counts = OperationCounts::measure(interpreters, out_dir, &format!("op-{}", budget.name), &script)?;
        operations.push(Operation {
            budget,
            counts: counts.per_operation(&empty),
        });
    }
    Ok(operations)
}

impl OperationCounts {
    fn measure(interpreters: &Interpreters, out_dir: &Path, label: &str, script: &Path) -> Result<Self, String> {
        let run = |program: &str, options: &[&str], kind: &str| measure(program, options, out_dir, &format!("{kind}-{label}"), script, &[]);
        Ok(Self {
            puc: run(&interpreters.puc, &[], "puc")?,
            interpreter: run(&interpreters.fazo, INTERPRETER_ONLY, "interpreter")?,
            baseline: run(&interpreters.fazo, BASELINE_ONLY, "baseline")?,
            peak: run(&interpreters.fazo, PEAK_ONLY, "peak")?,
            luajit: run(&interpreters.luajit, &[], "luajit")?,
        })
    }

    fn per_operation(&self, empty: &Self) -> Self {
        Self {
            puc: per_operation(self.puc, empty.puc),
            interpreter: per_operation(self.interpreter, empty.interpreter),
            baseline: per_operation(self.baseline, empty.baseline),
            peak: per_operation(self.peak, empty.peak),
            luajit: per_operation(self.luajit, empty.luajit),
        }
    }
}

fn per_operation(instructions: f64, empty: f64) -> f64 {
    (instructions - empty) / OPERATIONS_PER_RUN
}

fn build_interpreters(source: &Path) -> Result<Interpreters, String> {
    build_puc(source)?;
    build_fazo()?;
    let luajit = build_luajit()?;
    let build = build_dir();
    Ok(Interpreters {
        puc: format!("{build}/bin/lua"),
        fazo: format!("{build}/target/profiling/fazo"),
        luajit,
    })
}

fn build_dir() -> String {
    match env::var_os("FAZO_PERF_BUILD_DIR") {
        Some(dir) => guest_path(Path::new(&dir)),
        None => DEFAULT_BUILD_DIR.to_owned(),
    }
}

fn build_puc(source: &Path) -> Result<(), String> {
    let src = guest_path(&source.join("src"));
    let build = build_dir();
    let script = format!(
        "set -e; mkdir -p {build}/bin; cd {src}; \
         gcc -std=gnu99 -O2 -Wall -Wextra -DLUA_USE_LINUX '-Dluai_makeseed()=0' \
         $(ls *.c | grep -v '^luac.c$') -o {build}/bin/lua -lm -ldl -Wl,-E"
    );
    run_shell(&script, "building PUC Lua in Linux")
}

fn build_fazo() -> Result<(), String> {
    let cli = guest_path(&Path::new(CRATE_DIR).join("..").join("fazo_cli"));
    let build = build_dir();
    let script = format!("set -e; cd {cli}; RUSTC_WRAPPER= CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR={build}/target cargo build --profile profiling");
    run_shell(&script, "building fazo in Linux")
}

/// `git archive` takes the pinned commit whatever the checkout has checked out, and writes nothing into it.
fn build_luajit() -> Result<String, String> {
    let build = build_dir();
    let tree = format!("{build}/luajit-{LUAJIT_COMMIT}");
    let (repository, fetch) = luajit_repository(&build);
    let script = format!(
        "set -eo pipefail; {fetch}; \
         if [ ! -d {tree} ]; then \
           rm -rf {tree}.part; mkdir -p {tree}.part; \
           git -c safe.directory={repository} -C {repository} archive {LUAJIT_COMMIT} | tar -x -C {tree}.part; \
           mv {tree}.part {tree}; \
         fi; \
         make -C {tree} -j8 amalg"
    );
    run_shell(&script, "building LuaJIT in Linux")?;
    Ok(format!("{tree}/src/luajit"))
}

fn luajit_repository(build: &str) -> (String, String) {
    if let Some(dir) = env::var_os(LUAJIT_DIR_VARIABLE) {
        return (guest_path(Path::new(&dir)), "true".to_owned());
    }
    let repository = format!("{build}/luajit-git");
    let fetch = format!(
        "if ! git -C {repository} cat-file -e {LUAJIT_COMMIT}^{{commit}} 2>/dev/null; then \
           git init -q {repository}; \
           git -C {repository} fetch -q --depth 1 {LUAJIT_REPOSITORY} {LUAJIT_COMMIT}; \
         fi"
    );
    (repository, fetch)
}

fn run_shell(script: &str, what: &str) -> Result<(), String> {
    let output = host_command("bash", &["-lc".to_owned(), script.to_owned()])
        .output()
        .map_err(|error| format!("{what}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!("{what} failed with {}: {}", output.status, combined(&output)))
}

fn check_same_output(interpreters: &Interpreters, name: &str, scripts: &Scripts) -> Result<(), String> {
    let puc = plain_run(&interpreters.puc, &[], scripts.fazo, scripts.arguments)?;
    let mut runs: Vec<(&str, &str, &[&str], &Path)> = vec![
        ("fazo", &interpreters.fazo, &[], scripts.fazo),
        ("fazo's interpreter", &interpreters.fazo, INTERPRETER_ONLY, scripts.fazo),
    ];
    if let Some(script) = scripts.luajit {
        runs.push(("LuaJIT", &interpreters.luajit, &[], script));
        runs.push(("LuaJIT -joff", &interpreters.luajit, LUAJIT_JIT_OFF, script));
    }

    for (what, program, options, script) in runs {
        if plain_run(program, options, script, scripts.arguments)? != puc {
            return Err(format!("{name} differs between PUC Lua and {what}"));
        }
    }
    Ok(())
}

fn plain_run(interpreter: &str, options: &[&str], script: &Path, arguments: &[&str]) -> Result<(Vec<u8>, Option<i32>), String> {
    let output = host_command(interpreter, &arguments_for(options, script, arguments))
        .output()
        .map_err(|error| format!("starting {interpreter}: {error}"))?;
    Ok((output.stdout, output.status.code()))
}

fn measure(interpreter: &str, options: &[&str], out_dir: &Path, label: &str, script: &Path, arguments: &[&str]) -> Result<f64, String> {
    let out = out_dir.join(format!("{label}.out"));
    let mut args = vec![
        "--tool=cachegrind".to_owned(),
        "--cache-sim=no".to_owned(),
        format!("--cachegrind-out-file={}", guest_path(&out)),
        interpreter.to_owned(),
    ];
    args.extend(arguments_for(options, script, arguments));
    let output = host_command("valgrind", &args)
        .output()
        .map_err(|error| format!("starting valgrind: {error}"))?;
    if !output.status.success() {
        return Err(format!("valgrind on {label} failed with {}: {}", output.status, combined(&output)));
    }
    read_totals(&out)?
        .get("Ir")
        .copied()
        .ok_or_else(|| format!("cachegrind output for {label} has no Ir total"))
}

fn read_totals(path: &Path) -> Result<BTreeMap<String, f64>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    let names = text.lines().find_map(|line| line.strip_prefix("events: ")).ok_or("cachegrind output has no events line")?;
    let values = text.lines().find_map(|line| line.strip_prefix("summary: ")).ok_or("cachegrind output has no summary line")?;
    names
        .split_whitespace()
        .zip(values.split_whitespace())
        .map(|(name, value)| {
            value
                .parse::<f64>()
                .map(|value| (name.to_owned(), value))
                .map_err(|error| format!("reading the summary in {}: {error}", path.display()))
        })
        .collect()
}

fn arguments_for(options: &[&str], script: &Path, arguments: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = options.iter().map(|option| (*option).to_owned()).collect();
    args.push(guest_path(script));
    args.extend(arguments.iter().map(|argument| (*argument).to_owned()));
    args
}

fn slug(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect()
}

fn host_command(program: &str, args: &[String]) -> Command {
    if cfg!(windows) {
        let mut command = Command::new("wsl.exe");
        command.arg("-e").arg(program).args(args);
        return command;
    }
    let mut command = Command::new(program);
    command.args(args);
    command
}

fn guest_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    if !cfg!(windows) {
        return text;
    }
    match text.split_once(":/") {
        Some((drive, rest)) if drive.len() == 1 => format!("/mnt/{}/{rest}", drive.to_ascii_lowercase()),
        _ => text,
    }
}

fn combined(output: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{CRATE_DIR, LUAJIT_SUFFIX, WORKLOADS, guest_path};

    #[test]
    fn every_luajit_version_exists_and_is_used() {
        let dir = Path::new(CRATE_DIR).join("perf");
        let used: Vec<String> = WORKLOADS.iter().filter_map(|workload| workload.luajit_script()).collect();
        for script in &used {
            assert!(dir.join(script).is_file(), "{script} is missing");
        }
        for entry in fs::read_dir(&dir).expect("the perf folder is vendored") {
            let name = entry.expect("the perf folder reads").file_name().to_string_lossy().into_owned();
            if name.ends_with(LUAJIT_SUFFIX) {
                assert!(used.contains(&name), "no workload runs {name}");
            }
        }
    }

    #[test]
    fn windows_paths_reach_linux_as_mounts() {
        let expected = if cfg!(windows) { "/mnt/c/Users/me/Temp/fazo-perf" } else { "C:/Users/me/Temp/fazo-perf" };
        assert_eq!(guest_path(Path::new("C:/Users/me/Temp/fazo-perf")), expected);
        if cfg!(windows) {
            assert_eq!(guest_path(Path::new(r"C:\Users\me\Temp\fazo-perf")), expected);
        }
        assert_eq!(guest_path(Path::new("/tmp/fazo-perf")), "/tmp/fazo-perf");
    }
}
