use std::fmt::Write as _;
use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    process::{Command, Output},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use crate::{reference, report};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const DEFAULT_BUILD_DIR: &str = "/tmp/fazo-perf";
const EMPTY_SCRIPT: &str = "empty.lua";
const INTERPRETER_ONLY: &[&str] = &["-X", "tier=interpreter"];
const BASELINE_ONLY: &[&str] = &["-X", "tier=baseline"];
const PEAK_ONLY: &[&str] = &["-X", "force=peak"];
const OPTIMIZING_ONLY: &[&str] = &["-X", "force=optimizing"];
const OPERATIONS_DIR: &str = "ops";
const BUDGETS_FILE: &str = "budgets.txt";
const OPERATIONS_PER_RUN: f64 = 2_000_000.0;
const OPERATION_TOLERANCE: f64 = 1.0;
const LUAJIT_COMMIT: &str = "a4c064539ea931274d407e1249b4bb575424a0f4";
const LUAJIT_REPOSITORY: &str = "https://github.com/LuaJIT/LuaJIT.git";
const LUAJIT_DIR_VARIABLE: &str = "FAZO_LUAJIT_DIR";
const LUAJIT_JIT_OFF: &[&str] = &["-joff"];
const LUAJIT_SUFFIX: &str = ".luajit.lua";
const PARALLEL_RUNS: usize = 4;
const COMPILE_STATEMENTS: u32 = 500;
const COMPILE_CALLS: u32 = 200;
/// `aaa = aaa + 1` is GETTABUP, ADDI, MMBINI and SETTABUP in Lua 5.5, and GGET, ADDVN and GSET in LuaJIT (`luajit -bl`).
const FAZO_BYTECODES_PER_STATEMENT: f64 = 4.0;
const LUAJIT_BYTECODES_PER_STATEMENT: f64 = 3.0;
/// Callgrind counts only inside `Vm::jit_compile_optimized_at`, which every Optimizing and Peak compile goes through.
const OPTIMIZED_COMPILE_SYMBOL: &str = "*2Vm24jit_compile_optimized_at";
const PEAK_COMPILE_SYMBOLS: &[&str] = &["*4fazo3jit7compile7compile", "fazo::jit::compile::compile"];
const OPTIMIZING_COMPILE_SYMBOLS: &[&str] = &["*4fazo3jit7compile18compile_optimizing*", "fazo::jit::compile::compile_optimizing"];
const ALL_COMPILE_SYMBOLS: &[&str] = &[
    "*4fazo3jit7compile7compile",
    "fazo::jit::compile::compile",
    "*4fazo3jit7compile18compile_optimizing*",
    "fazo::jit::compile::compile_optimizing",
    "*4fazo3jit8baseline7compile*",
    "fazo::jit::baseline::compile",
];
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
    optimizing: f64,
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
    optimizing: f64,
    peak: f64,
    interpreter: f64,
    compile: f64,
    before_optimized: f64,
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
    check_outputs(&interpreters, &scripts)?;

    let mut plan = Plan::default();
    plan_workloads(&mut plan, &interpreters, &scripts);
    plan_operations(&mut plan, &interpreters, &operations_dir, &budgets);
    plan_compile_cost(&mut plan, &interpreters, &out_dir)?;
    let samples = plan.run(&out_dir)?;

    let measurements = workload_measurements(&samples)?;
    let operations = operation_measurements(&samples, &budgets)?;
    let compile_cost = compile_cost(&samples)?;
    let path = report::update(|board| {
        record(board, &measurements, &operations);
        board.compile_cost = Some(compile_cost);
    })?;
    print!("{}", summary(&measurements, &operations));
    println!("{}", compile_cost_line(&compile_cost));
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

#[derive(Clone, Copy)]
enum Probe {
    Total,
    Inside(&'static [&'static str]),
    Split,
}

struct Request {
    label: String,
    program: String,
    args: Vec<String>,
    probe: Probe,
}

#[derive(Clone, Copy)]
struct FazoSplit {
    total: f64,
    compile: f64,
    before_optimized: f64,
}

#[derive(Clone, Copy)]
enum Sample {
    Total(f64),
    Split(FazoSplit),
}

#[derive(Default)]
struct Plan {
    requests: Vec<Request>,
}

struct Samples {
    by_label: BTreeMap<String, Sample>,
}

type Slot = Option<Result<Sample, String>>;

impl Plan {
    fn add(&mut self, label: String, program: &str, args: Vec<String>, probe: Probe) {
        self.requests.push(Request {
            label,
            program: program.to_owned(),
            args,
            probe,
        });
    }

    /// Ir counts do not depend on load, so the runs go in parallel, at most `PARALLEL_RUNS` at once for the machine's thermal limit.
    fn run(self, out_dir: &Path) -> Result<Samples, String> {
        let next = AtomicUsize::new(0);
        let slots: Mutex<Vec<Slot>> = Mutex::new((0..self.requests.len()).map(|_| None).collect());
        let requests = &self.requests;

        thread::scope(|scope| {
            for _ in 0..PARALLEL_RUNS {
                scope.spawn(|| run_requests(requests, &next, &slots, out_dir));
            }
        });

        let slots = slots.into_inner().map_err(|_| "a perf run panicked".to_owned())?;
        let mut by_label = BTreeMap::new();
        for (request, slot) in self.requests.into_iter().zip(slots) {
            let sample = slot.ok_or_else(|| format!("{} did not run", request.label))??;
            by_label.insert(request.label, sample);
        }
        Ok(Samples { by_label })
    }
}

fn run_requests(requests: &[Request], next: &AtomicUsize, slots: &Mutex<Vec<Slot>>, out_dir: &Path) {
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(request) = requests.get(index) else {
            return;
        };
        let sample = request.sample(out_dir);
        if let Ok(mut slots) = slots.lock() {
            slots[index] = Some(sample);
        }
    }
}

impl Request {
    fn sample(&self, out_dir: &Path) -> Result<Sample, String> {
        match self.probe {
            Probe::Total => measure(&self.program, &self.args, out_dir, &self.label).map(Sample::Total),
            Probe::Inside(symbols) => inside(&self.program, &self.args, symbols, out_dir, &self.label).map(Sample::Total),
            Probe::Split => split(&self.program, &self.args, out_dir, &self.label).map(Sample::Split),
        }
    }
}

impl Samples {
    fn total(&self, label: &str) -> Result<f64, String> {
        match self.by_label.get(label) {
            Some(Sample::Total(value)) => Ok(*value),
            Some(Sample::Split(split)) => Ok(split.total),
            None => Err(format!("no measurement for {label}")),
        }
    }

    fn split(&self, label: &str) -> Result<FazoSplit, String> {
        match self.by_label.get(label) {
            Some(Sample::Split(split)) => Ok(*split),
            _ => Err(format!("no split measurement for {label}")),
        }
    }
}

fn check_outputs(interpreters: &Interpreters, scripts: &Path) -> Result<(), String> {
    for workload in WORKLOADS {
        let script = scripts.join(workload.script);
        let luajit_script = workload.luajit_script().map(|name| scripts.join(name));
        let inputs = Scripts {
            fazo: &script,
            luajit: luajit_script.as_deref(),
            arguments: workload.arguments,
        };
        check_same_output(interpreters, workload.name, &inputs)?;
    }
    Ok(())
}

fn plan_workloads(plan: &mut Plan, interpreters: &Interpreters, scripts: &Path) {
    let empty_script = scripts.join(EMPTY_SCRIPT);
    let empty = Scripts {
        fazo: &empty_script,
        luajit: Some(&empty_script),
        arguments: &[],
    };

    plan_counts(plan, interpreters, "empty", &empty);
    for workload in WORKLOADS {
        let script = scripts.join(workload.script);
        let luajit_script = workload.luajit_script().map(|name| scripts.join(name));
        let inputs = Scripts {
            fazo: &script,
            luajit: luajit_script.as_deref(),
            arguments: workload.arguments,
        };
        plan_counts(plan, interpreters, &slug(workload.name), &inputs);
    }
}

const WORKLOAD_RUNS: [(&str, &[&str]); 5] = [
    ("puc", &[]),
    ("baseline", BASELINE_ONLY),
    ("optimizing", OPTIMIZING_ONLY),
    ("peak", PEAK_ONLY),
    ("interpreter", INTERPRETER_ONLY),
];

fn plan_counts(plan: &mut Plan, interpreters: &Interpreters, label: &str, scripts: &Scripts) {
    let script = scripts.fazo;

    plan.add(format!("fazo-{label}"), &interpreters.fazo, arguments_for(&[], script, scripts.arguments), Probe::Split);
    for (kind, options) in WORKLOAD_RUNS {
        let program = if kind == "puc" { &interpreters.puc } else { &interpreters.fazo };
        plan.add(format!("{kind}-{label}"), program, arguments_for(options, script, scripts.arguments), Probe::Total);
    }
    let Some(luajit) = scripts.luajit else {
        return;
    };
    plan.add(format!("luajit-{label}"), &interpreters.luajit, arguments_for(&[], luajit, scripts.arguments), Probe::Total);
    plan.add(
        format!("luajit-interpreter-{label}"),
        &interpreters.luajit,
        arguments_for(LUAJIT_JIT_OFF, luajit, scripts.arguments),
        Probe::Total,
    );
}

fn workload_measurements(samples: &Samples) -> Result<Vec<Measurement>, String> {
    let empty = Counts::read(samples, "empty", true)?;
    WORKLOADS
        .iter()
        .map(|workload| {
            let counts = Counts::read(samples, &slug(workload.name), workload.luajit_script().is_some())?;
            Ok(Measurement {
                workload,
                counts: counts.minus(&empty),
            })
        })
        .collect()
}

impl Counts {
    fn read(samples: &Samples, label: &str, has_luajit: bool) -> Result<Self, String> {
        let total = |kind: &str| samples.total(&format!("{kind}-{label}"));
        let split = samples.split(&format!("fazo-{label}"))?;
        let luajit = if has_luajit {
            Some(LuajitCounts {
                jit: total("luajit")?,
                interpreter: total("luajit-interpreter")?,
            })
        } else {
            None
        };

        Ok(Self {
            puc: total("puc")?,
            fazo: split.total,
            baseline: total("baseline")?,
            optimizing: total("optimizing")?,
            peak: total("peak")?,
            interpreter: total("interpreter")?,
            compile: split.compile,
            before_optimized: split.before_optimized,
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
            optimizing: self.optimizing - empty.optimizing,
            peak: self.peak - empty.peak,
            interpreter: self.interpreter - empty.interpreter,
            compile: self.compile - empty.compile,
            before_optimized: self.before_optimized - empty.before_optimized,
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
        optimizing: Some(whole(counts.optimizing)),
        peak: whole(counts.peak),
        interpreter: whole(counts.interpreter),
        split: Some(counts.split()),
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
        optimizing: tenths(counts.optimizing),
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
            "{}: PUC {:.1}, interpreter {:.1}, baseline {:.1}, optimizing {:.1}, peak {:.1}, LuaJIT {:.1}, budget {:.1}",
            operation.budget.name,
            counts.puc,
            counts.interpreter,
            counts.baseline,
            counts.optimizing,
            counts.peak,
            counts.luajit,
            operation.budget.budget_ir,
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
    let split = counts.split();
    format!(
        "{}: PUC {:.0}, fazo {:.0}, baseline {:.0}, optimizing {:.0}, peak {:.0}, interpreter {:.0}, fazo/PUC {:.3}, {luajit}, \
         split compile {} before optimized {} steady {}",
        measurement.workload.name,
        counts.puc,
        counts.fazo,
        counts.baseline,
        counts.optimizing,
        counts.peak,
        counts.interpreter,
        measurement.instruction_ratio(),
        split.compile,
        split.before_optimized,
        split.steady,
    )
}

impl Counts {
    fn split(&self) -> report::Split {
        report::Split {
            compile: whole(self.compile),
            before_optimized: whole(self.before_optimized),
            steady: whole(self.fazo - self.compile - self.before_optimized),
        }
    }
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

const OPERATION_RUNS: [(&str, &[&str]); 5] = [
    ("puc", &[]),
    ("interpreter", INTERPRETER_ONLY),
    ("baseline", BASELINE_ONLY),
    ("optimizing", OPTIMIZING_ONLY),
    ("peak", PEAK_ONLY),
];

fn plan_operations(plan: &mut Plan, interpreters: &Interpreters, dir: &Path, budgets: &[Budget]) {
    plan_operation(plan, interpreters, "op-empty", &dir.join(EMPTY_SCRIPT));
    for budget in budgets {
        plan_operation(plan, interpreters, &format!("op-{}", budget.name), &dir.join(format!("{}.lua", budget.name)));
    }
}

fn plan_operation(plan: &mut Plan, interpreters: &Interpreters, label: &str, script: &Path) {
    for (kind, options) in OPERATION_RUNS {
        let program = if kind == "puc" { &interpreters.puc } else { &interpreters.fazo };
        plan.add(format!("{kind}-{label}"), program, arguments_for(options, script, &[]), Probe::Total);
    }
    plan.add(format!("luajit-{label}"), &interpreters.luajit, arguments_for(&[], script, &[]), Probe::Total);
}

fn operation_measurements<'a>(samples: &Samples, budgets: &'a [Budget]) -> Result<Vec<Operation<'a>>, String> {
    let empty = OperationCounts::read(samples, "op-empty")?;
    budgets
        .iter()
        .map(|budget| {
            let counts = OperationCounts::read(samples, &format!("op-{}", budget.name))?;
            Ok(Operation {
                budget,
                counts: counts.per_operation(&empty),
            })
        })
        .collect()
}

impl OperationCounts {
    fn read(samples: &Samples, label: &str) -> Result<Self, String> {
        let total = |kind: &str| samples.total(&format!("{kind}-{label}"));
        Ok(Self {
            puc: total("puc")?,
            interpreter: total("interpreter")?,
            baseline: total("baseline")?,
            optimizing: total("optimizing")?,
            peak: total("peak")?,
            luajit: total("luajit")?,
        })
    }

    fn per_operation(&self, empty: &Self) -> Self {
        Self {
            puc: per_operation(self.puc, empty.puc),
            interpreter: per_operation(self.interpreter, empty.interpreter),
            baseline: per_operation(self.baseline, empty.baseline),
            optimizing: per_operation(self.optimizing, empty.optimizing),
            peak: per_operation(self.peak, empty.peak),
            luajit: per_operation(self.luajit, empty.luajit),
        }
    }
}

fn per_operation(instructions: f64, empty: f64) -> f64 {
    (instructions - empty) / OPERATIONS_PER_RUN
}

/// The difference between a chunk with the statements and one without leaves the hot function's own cost. fazo's is the Ir
/// spent inside Peak compiles under `-X force=peak`; LuaJIT's, as the brief that set the target measured it, is its run with
/// the JIT on less its run with `-joff`, which also credits LuaJIT the speedup of the 200 calls.
fn plan_compile_cost(plan: &mut Plan, interpreters: &Interpreters, out_dir: &Path) -> Result<(), String> {
    let full = write_compile_chunk(out_dir, COMPILE_STATEMENTS)?;
    let empty = write_compile_chunk(out_dir, 0)?;

    for (suffix, script) in [("", &full), ("-empty", &empty)] {
        let options = |options: &[&str]| arguments_for(options, script, &[]);
        plan.add(format!("compile-peak{suffix}"), &interpreters.fazo, options(PEAK_ONLY), Probe::Inside(PEAK_COMPILE_SYMBOLS));
        plan.add(
            format!("compile-optimizing{suffix}"),
            &interpreters.fazo,
            options(OPTIMIZING_ONLY),
            Probe::Inside(OPTIMIZING_COMPILE_SYMBOLS),
        );
        plan.add(format!("compile-luajit{suffix}"), &interpreters.luajit, options(&[]), Probe::Total);
        plan.add(format!("compile-luajit{suffix}-off"), &interpreters.luajit, options(LUAJIT_JIT_OFF), Probe::Total);
    }
    Ok(())
}

fn compile_cost(samples: &Samples) -> Result<report::CompileCost, String> {
    let difference = |label: &str| -> Result<f64, String> { Ok(samples.total(label)? - samples.total(&format!("{label}-empty"))?) };
    let peak_ir = difference("compile-peak")?;
    let optimizing_ir = difference("compile-optimizing")?;
    let luajit_ir = (samples.total("compile-luajit")? - samples.total("compile-luajit-off")?)
        - (samples.total("compile-luajit-empty")? - samples.total("compile-luajit-empty-off")?);
    let statements = f64::from(COMPILE_STATEMENTS);

    Ok(report::CompileCost {
        statements: COMPILE_STATEMENTS,
        peak_per_bytecode: tenths(peak_ir / (statements * FAZO_BYTECODES_PER_STATEMENT)),
        optimizing_per_bytecode: Some(tenths(optimizing_ir / (statements * FAZO_BYTECODES_PER_STATEMENT))),
        luajit_per_bytecode: tenths(luajit_ir / (statements * LUAJIT_BYTECODES_PER_STATEMENT)),
        peak_per_statement: tenths(peak_ir / statements),
        luajit_per_statement: tenths(luajit_ir / statements),
    })
}

fn write_compile_chunk(out_dir: &Path, statements: u32) -> Result<std::path::PathBuf, String> {
    let mut source = String::from("aaa = 0\nlocal function f()\n");
    for _ in 0..statements {
        source.push_str("aaa = aaa + 1\n");
    }
    let _ = write!(source, "end\nfor i = 1, {COMPILE_CALLS} do f() end\nprint(aaa)\n");
    let path = out_dir.join(format!("compile-{statements}.lua"));
    fs::write(&path, source).map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(path)
}

fn compile_cost_line(cost: &report::CompileCost) -> String {
    let optimizing = cost.optimizing_per_bytecode.unwrap_or_default();
    format!(
        "compile cost ({} statements): Peak {:.0} Ir per bytecode ({:.0} per statement), Optimizing {:.0}, LuaJIT {:.0} ({:.0}), \
         Peak/LuaJIT {:.2}, Optimizing/LuaJIT {:.2} per bytecode",
        cost.statements,
        cost.peak_per_bytecode,
        cost.peak_per_statement,
        optimizing,
        cost.luajit_per_bytecode,
        cost.luajit_per_statement,
        cost.peak_per_bytecode / cost.luajit_per_bytecode,
        optimizing / cost.luajit_per_bytecode,
    )
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

fn measure(interpreter: &str, arguments: &[String], out_dir: &Path, label: &str) -> Result<f64, String> {
    let out = out_dir.join(format!("{label}.out"));
    let mut args = vec![
        "--tool=cachegrind".to_owned(),
        "--cache-sim=no".to_owned(),
        format!("--cachegrind-out-file={}", guest_path(&out)),
        interpreter.to_owned(),
    ];
    args.extend_from_slice(arguments);

    valgrind(&args, label)?;
    read_totals(&out)?
        .get("Ir")
        .copied()
        .ok_or_else(|| format!("cachegrind output for {label} has no Ir total"))
}

fn valgrind(args: &[String], label: &str) -> Result<(), String> {
    let output = host_command("valgrind", args)
        .output()
        .map_err(|error| format!("starting valgrind: {error}"))?;
    if !output.status.success() {
        return Err(format!("valgrind on {label} failed with {}: {}", output.status, combined(&output)));
    }
    Ok(())
}

/// One callgrind run gives fazo's total, the Ir inside compiles and, because callgrind dumps before each
/// `OPTIMIZED_COMPILE_SYMBOL` call, the work before the first compile in its first part.
fn split(interpreter: &str, arguments: &[String], out_dir: &Path, label: &str) -> Result<FazoSplit, String> {
    let out = out_dir.join(format!("{label}.callgrind"));
    let mut args = vec![
        "--tool=callgrind".to_owned(),
        format!("--callgrind-out-file={}", guest_path(&out)),
        format!("--dump-before={OPTIMIZED_COMPILE_SYMBOL}"),
        interpreter.to_owned(),
    ];
    args.extend_from_slice(arguments);

    remove_parts(out_dir, label)?;
    valgrind(&args, label)?;
    let mut parts: Vec<_> = (1..)
        .map(|part| out_dir.join(format!("{label}.callgrind.{part}")))
        .take_while(|path| path.exists())
        .collect();
    parts.push(out);

    let mut names = BTreeMap::new();
    let mut totals = Vec::with_capacity(parts.len());
    let mut compile = 0.0;
    for path in &parts {
        let text = fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
        let part = callgrind_part(&text, &mut names).ok_or_else(|| format!("callgrind output for {label} has no total"))?;
        totals.push(part.total);
        compile += part.compile;
    }
    Ok(FazoSplit {
        total: totals.iter().sum(),
        compile,
        before_optimized: totals[0],
    })
}

fn remove_parts(out_dir: &Path, label: &str) -> Result<(), String> {
    let prefix = format!("{label}.callgrind.");
    let entries = fs::read_dir(out_dir).map_err(|error| format!("reading {}: {error}", out_dir.display()))?;
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            fs::remove_file(entry.path()).map_err(|error| format!("removing {}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

struct CallgrindPart {
    total: f64,
    compile: f64,
}

/// The compile share sums the inclusive cost of calls from outside the compile functions into them, so nested compiles count once.
fn callgrind_part(text: &str, names: &mut BTreeMap<String, String>) -> Option<CallgrindPart> {
    let mut total = None;
    let mut compile = 0.0;
    let mut caller_compiles = false;
    let mut callee_compiles = false;
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("fn=") {
            caller_compiles = is_compile(&function_name(rest, names));
        } else if let Some(rest) = line.strip_prefix("cfn=") {
            callee_compiles = is_compile(&function_name(rest, names));
        } else if line.starts_with("calls=") {
            let cost = lines.next().and_then(|cost| cost.split_whitespace().nth(1)?.parse::<f64>().ok()).unwrap_or(0.0);
            if callee_compiles && !caller_compiles {
                compile += cost;
            }
        } else if let Some(rest) = line.strip_prefix("totals: ").or_else(|| line.strip_prefix("summary: ")) {
            total = total.or_else(|| rest.split_whitespace().next()?.parse::<f64>().ok());
        }
    }
    Some(CallgrindPart { total: total?, compile })
}

fn function_name(field: &str, names: &mut BTreeMap<String, String>) -> String {
    let Some((id, name)) = field.strip_prefix('(').and_then(|rest| rest.split_once(')')) else {
        return field.to_owned();
    };
    let name = name.trim_start();
    if name.is_empty() {
        return names.get(id).cloned().unwrap_or_default();
    }
    names.insert(id.to_owned(), name.to_owned());
    name.to_owned()
}

fn is_compile(name: &str) -> bool {
    ALL_COMPILE_SYMBOLS.iter().any(|pattern| glob_matches(pattern, name))
}

fn glob_matches(pattern: &str, text: &str) -> bool {
    let pieces: Vec<&str> = pattern.split('*').collect();
    let Some((first, rest)) = pieces.split_first() else {
        return text.is_empty();
    };
    let Some(mut remaining) = text.strip_prefix(first) else {
        return false;
    };
    let Some((last, middle)) = rest.split_last() else {
        return remaining.is_empty();
    };
    for piece in middle {
        let Some(at) = remaining.find(piece) else {
            return false;
        };
        remaining = &remaining[at + piece.len()..];
    }
    remaining.ends_with(last)
}

fn callgrind_total(path: &Path, label: &str) -> Result<f64, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    text.lines()
        .find_map(|line| line.strip_prefix("totals: ").or_else(|| line.strip_prefix("summary: ")))
        .and_then(|value| value.split_whitespace().next()?.parse::<f64>().ok())
        .ok_or_else(|| format!("callgrind output for {label} has no total"))
}

fn inside(interpreter: &str, arguments: &[String], symbols: &[&str], out_dir: &Path, label: &str) -> Result<f64, String> {
    let out = out_dir.join(format!("{label}.callgrind"));
    let mut args = vec!["--tool=callgrind".to_owned(), format!("--callgrind-out-file={}", guest_path(&out))];
    args.extend(symbols.iter().map(|symbol| format!("--toggle-collect={symbol}")));
    args.push(interpreter.to_owned());
    args.extend_from_slice(arguments);

    valgrind(&args, label)?;
    callgrind_total(&out, label)
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

    use std::collections::BTreeMap;

    use super::{CRATE_DIR, LUAJIT_SUFFIX, WORKLOADS, callgrind_part, glob_matches, guest_path};

    #[test]
    fn globs_match_mangled_symbols() {
        assert!(glob_matches("*4fazo3jit7compile7compile", "_RNvNtNtCs1_4fazo3jit7compile7compile"));
        assert!(!glob_matches("*4fazo3jit7compile7compile", "_RNvNtNtCs1_4fazo3jit7compile7compile2"));
        assert!(glob_matches("*4fazo3jit8baseline7compile*", "_RNvNtNtCs1_4fazo3jit8baseline7compile.llvm.1"));
        assert!(glob_matches("fazo::jit::compile::compile", "fazo::jit::compile::compile"));
    }

    #[test]
    fn callgrind_parts_count_compile_calls_once() {
        let text = "events: Ir\nsummary: 1000\n\
                    fn=(1) main\n5 10\ncfn=(2) _RNvNtNtCs1_4fazo3jit7compile7compile\ncalls=1 3\n6 300\n\
                    fn=(2)\n7 20\ncfn=(3) _RNvNtNtCs1_4fazo3jit8baseline7compile\ncalls=1 3\n8 50\n\
                    fn=(1)\ncfn=(3)\ncalls=2 3\n9 70\n";
        let mut names = BTreeMap::new();
        let part = callgrind_part(text, &mut names).expect("the part has a total");
        assert_eq!(part.total, 1000.0);
        assert_eq!(part.compile, 370.0);
    }

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
