use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{
    differential::{self, JitMode, Outcome},
    generator::{self, Profile},
    official, perf, reference,
};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const WORKLOAD_DEADLINE: Duration = Duration::from_secs(120);
const OFFICIAL_DEADLINE: Duration = Duration::from_secs(300);
const OFFICIAL_FILES: [&str; 5] = ["math.lua", "calls.lua", "nextvar.lua", "coroutine.lua", "sort.lua"];
const INTERPRETER: &[&str] = &["-X", "tier=interpreter"];
const MODES: [(&str, &[&str]); 3] = [
    ("force=peak", &["-X", "force=peak", "-X", "jitstats"]),
    ("force=peak stress", &["-X", "force=peak", "-X", "stress-exit=3", "-X", "stress-ic", "-X", "jitstats"]),
    ("default", &["-X", "jitstats"]),
];
/// A storm counts even when output is right; a recursive prototype deoptimizes once per retired frame, so the site limit sits above the recompile limit.
const SITE_DEOPTS_LIMIT: u32 = 8;
const PEAK_COMPILES_LIMIT: u32 = 6;
const RECOMPILES_LIMIT: u32 = 8;

#[derive(Default)]
pub struct Report {
    pub failures: Vec<String>,
    pub storms: Vec<String>,
    pub lines: Vec<String>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.failures.is_empty() && self.storms.is_empty()
    }
}

pub fn run(fazo: &Path, seeds: u64) -> Report {
    let mut report = Report::default();
    let started = Instant::now();
    workloads(fazo, &mut report);
    report.lines.push(format!("workloads: {:.1}s", started.elapsed().as_secs_f64()));
    let started = Instant::now();
    operations(fazo, &mut report);
    report.lines.push(format!("operations: {:.1}s", started.elapsed().as_secs_f64()));
    let started = Instant::now();
    generated(seeds, &mut report);
    report.lines.push(format!("seeds: {:.1}s", started.elapsed().as_secs_f64()));
    let started = Instant::now();
    official_files(fazo, &mut report);
    report.lines.push(format!("official files: {:.1}s", started.elapsed().as_secs_f64()));
    report
}

fn perf_dir() -> PathBuf {
    Path::new(CRATE_DIR).join("perf")
}

fn workloads(fazo: &Path, report: &mut Report) {
    let dir = perf_dir();
    for workload in perf::WORKLOADS {
        let expected = match run_workload(fazo, &dir, workload, INTERPRETER) {
            Ok(captured) => captured.stdout,
            Err(error) => {
                report.failures.push(format!("{} (interpreter): {error}", workload.name));
                continue;
            }
        };
        for (mode, flags) in MODES {
            let captured = match run_workload(fazo, &dir, workload, flags) {
                Ok(captured) => captured,
                Err(error) => {
                    report.failures.push(format!("{} ({mode}): {error}", workload.name));
                    continue;
                }
            };
            if captured.stdout != expected {
                report.failures.push(format!("{} ({mode}): output differs from the interpreter", workload.name));
            }
            let (summary, storms) = jit_stats(&captured.stderr);
            report.lines.push(format!("{:<34} {mode:<18} {summary}", workload.name));
            report.storms.extend(storms.into_iter().map(|storm| format!("{} ({mode}): {storm}", workload.name)));
        }
    }
}

/// Per-operation scripts from `cargo xtask perf`, run under `-X force=peak`.
fn operations(fazo: &Path, report: &mut Report) {
    let dir = perf_dir().join("ops");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        report.failures.push(format!("cannot read {}", dir.display()));
        return;
    };
    let mut scripts: Vec<String> = entries.filter_map(|entry| entry.ok()?.file_name().into_string().ok()).filter(|name| name.ends_with(".lua")).collect();
    scripts.sort();
    for script in scripts {
        let run = |flags: &[&str]| {
            let args: Vec<&str> = flags.iter().copied().chain([script.as_str()]).collect();
            reference::execute(fazo, &args, &dir, reference::scaled(WORKLOAD_DEADLINE))
        };
        let (Ok(expected), Ok(forced)) = (run(INTERPRETER), run(&["-X", "force=peak"])) else {
            report.failures.push(format!("ops/{script}: could not run"));
            continue;
        };
        if !forced.success || forced.stdout != expected.stdout {
            report.failures.push(format!("ops/{script} (force=peak): {}", String::from_utf8_lossy(reference::first_line(&forced.stderr))));
        }
    }
}

struct Captured {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_workload(fazo: &Path, dir: &Path, workload: &perf::Workload, flags: &[&str]) -> Result<Captured, String> {
    let mut args: Vec<&str> = flags.to_vec();
    args.push(workload.script);
    args.extend_from_slice(workload.arguments);
    let captured = reference::execute(fazo, &args, dir, reference::scaled(WORKLOAD_DEADLINE))?;
    if !captured.success {
        return Err(String::from_utf8_lossy(reference::first_line(&captured.stderr)).into_owned());
    }
    Ok(Captured {
        stdout: captured.stdout,
        stderr: captured.stderr,
    })
}

fn jit_stats(stderr: &[u8]) -> (String, Vec<String>) {
    let text = String::from_utf8_lossy(stderr);
    let mut totals = [0u32; 4];
    let mut storms = Vec::new();
    for line in text.lines().filter_map(|line| line.strip_prefix("jitstats ")) {
        let Some((name, fields)) = line.split_once(' ') else {
            continue;
        };
        let field = |key: &str| fields.split(' ').find_map(|part| part.strip_prefix(key)).unwrap_or("");
        let number = |key: &str| field(key).parse::<u32>().unwrap_or(0);
        let counts = [number("peak="), number("deopts="), number("recompiles="), number("declines=")];
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += count;
        }
        let hot_sites: Vec<&str> = field("sites=")
            .split(',')
            .filter(|site| site.split_once(':').and_then(|(_, count)| count.parse::<u32>().ok()).is_some_and(|count| count >= SITE_DEOPTS_LIMIT))
            .collect();
        if counts[0] >= PEAK_COMPILES_LIMIT || counts[2] >= RECOMPILES_LIMIT || !hot_sites.is_empty() {
            storms.push(format!("{name} {fields}"));
        }
    }
    let summary = format!("peak {} deopts {} recompiles {} declines {}", totals[0], totals[1], totals[2], totals[3]);
    (summary, storms)
}

fn generated(seeds: u64, report: &mut Report) {
    let mut checked = 0;
    for profile in [Profile::Full, Profile::Metatables, Profile::Core] {
        for seed in 0..seeds {
            let source = generator::program_for(seed, profile);
            differential::set_jit_mode(JitMode::Interpreter);
            let expected = differential::run_fazo(source.as_bytes(), "t");
            differential::set_jit_mode(JitMode::Peak);
            let peak = differential::run_fazo(source.as_bytes(), "t");
            checked += 1;
            if !same(&expected, &peak) {
                report.failures.push(format!("seed {profile:?}/{seed}: force=peak differs from the interpreter"));
            }
        }
    }
    differential::set_jit_mode(JitMode::Normal);
    report.lines.push(format!("generated programs: {checked} under force=peak"));
}

fn same(left: &Outcome, right: &Outcome) -> bool {
    left.output == right.output && left.error == right.error
}

fn official_files(fazo: &Path, report: &mut Report) {
    let suite = match reference::shared() {
        Ok(reference) => reference.suite.clone(),
        Err(error) => {
            report.failures.push(format!("official suite: {error}"));
            return;
        }
    };
    for name in OFFICIAL_FILES {
        let args = ["-X", "force=peak", "-e", official::PORTABLE_MODE, name];
        let verdict = match reference::execute(fazo, &args, &suite, reference::scaled(OFFICIAL_DEADLINE)) {
            Ok(captured) if captured.success => None,
            Ok(captured) => Some(String::from_utf8_lossy(&reference::error_message(fazo, &captured)).into_owned()),
            Err(error) => Some(error),
        };
        match verdict {
            None => report.lines.push(format!("official {name}: pass")),
            Some(reason) => report.failures.push(format!("official {name} (force=peak): {reason}")),
        }
    }
}

pub fn render(report: &Report) -> String {
    let mut text = String::new();
    for line in &report.lines {
        let _ = writeln!(text, "{line}");
    }
    for storm in &report.storms {
        let _ = writeln!(text, "STORM {storm}");
    }
    for failure in &report.failures {
        let _ = writeln!(text, "FAIL {failure}");
    }
    let _ = writeln!(text, "quick: {} failures, {} storms", report.failures.len(), report.storms.len());
    text
}
