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

pub struct Workload {
    pub name: &'static str,
    pub script: &'static str,
    pub arguments: &'static [&'static str],
}

pub const WORKLOADS: &[Workload] = &[
    Workload { name: "fib", script: "fib.lua", arguments: &[] },
    Workload { name: "integer loop", script: "int_loop.lua", arguments: &[] },
    Workload { name: "float loop", script: "float_loop.lua", arguments: &[] },
    Workload { name: "table array", script: "table_array.lua", arguments: &[] },
    Workload { name: "table hash", script: "table_hash.lua", arguments: &[] },
    Workload { name: "closures", script: "closures.lua", arguments: &[] },
    Workload { name: "method calls through __index", script: "method_index.lua", arguments: &[] },
    Workload { name: "absent metamethods", script: "absent_metamethods.lua", arguments: &[] },
    Workload { name: "coroutine switching", script: "coroutine_switch.lua", arguments: &[] },
    Workload { name: "strings", script: "strings.lua", arguments: &[] },
    Workload { name: "binary-trees", script: "binary_trees.lua", arguments: &[] },
    Workload { name: "n-body", script: "run_awfy.lua", arguments: &["nbody", "12000"] },
    Workload { name: "spectral-norm", script: "spectral_norm.lua", arguments: &[] },
    Workload { name: "fannkuch", script: "fannkuch.lua", arguments: &[] },
    Workload { name: "class dispatch", script: "class_dispatch.lua", arguments: &[] },
    Workload { name: "deltablue", script: "run_awfy.lua", arguments: &["deltablue", "1000"] },
    Workload { name: "havlak", script: "run_awfy.lua", arguments: &["havlak_sized", "1"] },
    Workload { name: "garbage collector (generational)", script: "gc_heap.lua", arguments: &[] },
    Workload { name: "stats", script: "stats.lua", arguments: &[] },
    Workload { name: "dialogue (interpreted)", script: "dialogue.lua", arguments: &["interpreted"] },
    Workload { name: "dialogue (compiled with load)", script: "dialogue.lua", arguments: &["compiled"] },
    Workload { name: "frame", script: "frame.lua", arguments: &[] },
];

struct Budget {
    name: String,
    budget_ir: f64,
    recorded_ir: f64,
    reason: String,
}

struct Operation<'a> {
    budget: &'a Budget,
    puc: f64,
    fazo: f64,
}

#[derive(Clone, Copy)]
struct Measurement {
    name: &'static str,
    puc: f64,
    fazo: f64,
    baseline: f64,
    peak: f64,
    interpreter: f64,
}

struct Interpreters {
    puc: String,
    fazo: String,
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
    let empty = scripts.join(EMPTY_SCRIPT);
    let empty_puc = measure(&interpreters.puc, &[], out_dir, "puc-empty", &empty, &[])?;
    let empty_fazo = measure(&interpreters.fazo, &[], out_dir, "fazo-empty", &empty, &[])?;
    let empty_interpreter = measure(&interpreters.fazo, INTERPRETER_ONLY, out_dir, "interpreter-empty", &empty, &[])?;
    let empty_baseline = measure(&interpreters.fazo, BASELINE_ONLY, out_dir, "baseline-empty", &empty, &[])?;
    let empty_peak = measure(&interpreters.fazo, PEAK_ONLY, out_dir, "peak-empty", &empty, &[])?;

    let mut measurements = Vec::with_capacity(WORKLOADS.len());
    for workload in WORKLOADS {
        let script = scripts.join(workload.script);
        let label = slug(workload.name);
        check_same_output(interpreters, workload, &script)?;
        let run = |interpreter: &str, options: &[&str], kind: &str| {
            measure(interpreter, options, out_dir, &format!("{kind}-{label}"), &script, workload.arguments)
        };
        measurements.push(Measurement {
            name: workload.name,
            puc: run(&interpreters.puc, &[], "puc")? - empty_puc,
            fazo: run(&interpreters.fazo, &[], "fazo")? - empty_fazo,
            baseline: run(&interpreters.fazo, BASELINE_ONLY, "baseline")? - empty_baseline,
            peak: run(&interpreters.fazo, PEAK_ONLY, "peak")? - empty_peak,
            interpreter: run(&interpreters.fazo, INTERPRETER_ONLY, "interpreter")? - empty_interpreter,
        });
    }
    Ok(measurements)
}

fn record(board: &mut report::Scoreboard, measurements: &[Measurement], operations: &[Operation]) {
    board.perf = measurements
        .iter()
        .map(|measurement| {
            let workload = report::Workload {
                puc: measurement.puc.round() as i64,
                fazo: measurement.fazo.round() as i64,
                baseline: measurement.baseline.round() as i64,
                peak: measurement.peak.round() as i64,
                interpreter: measurement.interpreter.round() as i64,
            };
            (measurement.name.to_owned(), workload)
        })
        .collect();
    board.operations = operations
        .iter()
        .map(|operation| {
            let entry = report::Operation {
                puc: tenths(operation.puc),
                fazo: tenths(operation.fazo),
                budget: tenths(operation.budget.budget_ir),
            };
            (operation.budget.name.clone(), entry)
        })
        .collect();
}

fn tenths(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn summary(measurements: &[Measurement], operations: &[Operation]) -> String {
    let mut text = String::new();
    for measurement in measurements {
        let _ = writeln!(
            text,
            "{}: PUC {:.0}, fazo {:.0}, baseline {:.0}, peak {:.0}, interpreter {:.0}, fazo/PUC {:.3}",
            measurement.name,
            measurement.puc,
            measurement.fazo,
            measurement.baseline,
            measurement.peak,
            measurement.interpreter,
            measurement.instruction_ratio(),
        );
    }
    for operation in operations {
        let _ = writeln!(
            text,
            "{}: PUC {:.1}, fazo {:.1}, budget {:.1}",
            operation.budget.name, operation.puc, operation.fazo, operation.budget.budget_ir,
        );
    }
    text
}

impl Measurement {
    fn instruction_ratio(&self) -> f64 {
        self.fazo / self.puc
    }
}

fn check_workloads(measurements: &[Measurement]) -> Result<(), String> {
    let failing: Vec<&str> = measurements
        .iter()
        .filter(|measurement| measurement.instruction_ratio() > 1.0)
        .map(|measurement| measurement.name)
        .collect();
    if failing.is_empty() {
        println!("workloads: every workload at or below PUC Lua");
        return Ok(());
    }
    Err(format!("workloads: above PUC Lua: {}", failing.join(", ")))
}

impl Operation<'_> {
    fn over_budget(&self) -> bool {
        self.budget.reason.is_empty() && self.fazo > self.budget.budget_ir + OPERATION_TOLERANCE
    }

    fn over_recorded(&self) -> bool {
        self.fazo > self.budget.recorded_ir + OPERATION_TOLERANCE
    }

    fn describe(&self) -> String {
        format!(
            "{} (fazo {:.1}, budget {:.1}, recorded {:.1})",
            self.budget.name, self.fazo, self.budget.budget_ir, self.budget.recorded_ir
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
    let empty = dir.join(EMPTY_SCRIPT);
    let puc_empty = measure(&interpreters.puc, &[], out_dir, "puc-op-empty", &empty, &[])?;
    let fazo_empty = measure(&interpreters.fazo, INTERPRETER_ONLY, out_dir, "interpreter-op-empty", &empty, &[])?;
    let mut operations = Vec::with_capacity(budgets.len());
    for budget in budgets {
        let script = dir.join(format!("{}.lua", budget.name));
        let puc = measure(&interpreters.puc, &[], out_dir, &format!("puc-op-{}", budget.name), &script, &[])?;
        let fazo = measure(
            &interpreters.fazo,
            INTERPRETER_ONLY,
            out_dir,
            &format!("interpreter-op-{}", budget.name),
            &script,
            &[],
        )?;
        operations.push(Operation {
            budget,
            puc: per_operation(puc, puc_empty),
            fazo: per_operation(fazo, fazo_empty),
        });
    }
    Ok(operations)
}

fn per_operation(instructions: f64, empty: f64) -> f64 {
    (instructions - empty) / OPERATIONS_PER_RUN
}

fn build_interpreters(source: &Path) -> Result<Interpreters, String> {
    build_puc(source)?;
    build_fazo()?;
    let build = build_dir();
    Ok(Interpreters {
        puc: format!("{build}/bin/lua"),
        fazo: format!("{build}/target/profiling/fazo"),
    })
}

fn build_dir() -> String {
    env::var("FAZO_PERF_BUILD_DIR").unwrap_or_else(|_| DEFAULT_BUILD_DIR.to_owned())
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

fn run_shell(script: &str, what: &str) -> Result<(), String> {
    let output = host_command("bash", &["-lc".to_owned(), script.to_owned()])
        .output()
        .map_err(|error| format!("{what}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!("{what} failed with {}: {}", output.status, combined(&output)))
}

fn check_same_output(interpreters: &Interpreters, workload: &Workload, script: &Path) -> Result<(), String> {
    let puc = plain_run(&interpreters.puc, &[], script, workload.arguments)?;
    let fazo = plain_run(&interpreters.fazo, &[], script, workload.arguments)?;
    if puc != fazo {
        return Err(format!("{} differs between PUC Lua and fazo", workload.name));
    }
    let interpreter = plain_run(&interpreters.fazo, INTERPRETER_ONLY, script, workload.arguments)?;
    if puc != interpreter {
        return Err(format!("{} differs between PUC Lua and fazo's interpreter", workload.name));
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
