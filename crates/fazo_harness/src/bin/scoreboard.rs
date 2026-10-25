use std::{
    env, fs,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use fazo_harness::{
    CASES_DIR,
    cli,
    differential::{self, Comparison},
    generator::{self, Profile},
    minimize::minimize,
    official, lua_files, reference,
    report::{self, Run, Suite},
};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const DEFAULT_SEEDS: u64 = 200;
const OFFICIAL_DEADLINE: Duration = Duration::from_secs(120);
const SYNTAX_DIR: &str = "syntax";
const STANDALONE_CASE: &str = "standalone main.lua";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Pass,
    KnownPucBug(&'static str),
    Fail(String),
}

impl Verdict {
    fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass | Verdict::KnownPucBug(_))
    }

    fn cell(&self) -> String {
        match self {
            Verdict::Pass => "pass".to_string(),
            Verdict::KnownPucBug(reason) => format!("pass: known PUC 5.5.1 bug, {reason}"),
            Verdict::Fail(reason) => format!("fail: {reason}"),
        }
    }
}

/// Cases where PUC Lua 5.5.1 has a bug that lua.org lists, so a mismatch is recorded rather than failed.
const KNOWN_PUC_BUGS: &[(&str, &str)] = &[
    ("float_minus_zero.lua", "(-0.0 - 0) should be -0.0 (lua.org bugs, 5.5.1)"),
    ("numeric_for_nan_limit.lua", "a NaN limit must not run the loop (lua.org bugs, 5.5.1)"),
];

enum Column {
    Ran(Verdict),
    NotRun,
}

struct OfficialRow {
    name: String,
    reference: Column,
    fazo: Column,
    parses: Verdict,
}

struct CaseRow {
    name: String,
    fazo: Verdict,
    parses: Verdict,
}

struct SyntaxRow {
    name: String,
    fazo: Verdict,
}

struct StandaloneRow {
    reference: Verdict,
    fazo: Verdict,
}

struct SeedReport {
    profile: Profile,
    seeds: u64,
    matched: u64,
    rejected: u64,
    mismatching_seeds: Vec<u64>,
}

/// Windows' 1 MiB main-thread stack is too small; Linux and macOS use 8 MiB.
const SCOREBOARD_STACK_BYTES: usize = 8 << 20;

fn main() -> ExitCode {
    let worker = std::thread::Builder::new().stack_size(SCOREBOARD_STACK_BYTES).spawn(run);
    match worker {
        Ok(handle) => handle.join().unwrap_or(ExitCode::FAILURE),
        Err(error) => fail(&format!("cannot start the scoreboard thread: {error}")),
    }
}

fn run() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let scale = match timeout_scale(&args) {
        Ok(scale) => scale,
        Err(message) => return fail(&message),
    };
    reference::set_timeout_scale(scale);
    if let Err(message) = jit_mode(&args) {
        return fail(&message);
    }
    let run = match entry(&args) {
        Ok(run) => run,
        Err(message) => return fail(&message),
    };
    match args.first().map(String::as_str) {
        Some("fetch") => fetch(),
        Some("mismatches") => mismatches(&args[1..]),
        Some("minimize") => minimize_seed(&args[1..]),
        _ => match seed_count(&args) {
            Ok(seeds) => scoreboard(seeds, run),
            Err(message) => {
                eprintln!(
                    "{message}\nusage: scoreboard [--seeds N] [--timeout-scale N] [--jit TIER] [--entry NAME] | scoreboard fetch \
                     | scoreboard mismatches full|core [--seeds N] | scoreboard minimize full|core SEED"
                );
                ExitCode::FAILURE
            }
        },
    }
}

fn fetch() -> ExitCode {
    let root = reference::cache_dir();
    match reference::fetch_into(&root) {
        Ok(()) => {
            println!("the reference sources are in {}", root.display());
            ExitCode::SUCCESS
        }
        Err(error) => fail(&error),
    }
}

fn mismatches(args: &[String]) -> ExitCode {
    let profile = match args.first().map(String::as_str) {
        Some("full") => Profile::Full,
        Some("metatables") => Profile::Metatables,
        Some("core") => Profile::Core,
        _ => return fail("usage: scoreboard mismatches full|metatables|core [--seeds N]"),
    };
    let seeds = match seed_count(args) {
        Ok(seeds) => seeds,
        Err(message) => return fail(&message),
    };
    for seed in 0..seeds {
        let source = generator::program_for(seed, profile);
        let comparison = match differential::compare(source.as_bytes()) {
            Ok(comparison) => comparison,
            Err(error) => return fail(&error),
        };
        if comparison.reference.error.is_some() || comparison.matches() {
            continue;
        }
        println!("{seed}\t{}", mismatch_signature(&comparison));
    }
    ExitCode::SUCCESS
}

fn minimize_seed(args: &[String]) -> ExitCode {
    let profile = match args.first().map(String::as_str) {
        Some("full") => Profile::Full,
        Some("metatables") => Profile::Metatables,
        Some("core") => Profile::Core,
        _ => return fail("usage: scoreboard minimize full|metatables|core SEED"),
    };
    let Some(seed) = args.get(1).and_then(|text| text.parse::<u64>().ok()) else {
        return fail("usage: scoreboard minimize full|core SEED");
    };
    let minimized = minimize(generator::lines_for(seed, profile), mismatch_persists);
    print!("{}", minimized.join("\n") + "\n");
    ExitCode::SUCCESS
}

fn mismatch_signature(comparison: &Comparison) -> String {
    if let Some(error) = &comparison.fazo.error {
        return format!("fazo error: {}", first_line(error));
    }
    let reference_text = String::from_utf8_lossy(&comparison.reference.output);
    let fazo_text = String::from_utf8_lossy(&comparison.fazo.output);
    let reference: Vec<&str> = reference_text.lines().collect();
    let fazo: Vec<&str> = fazo_text.lines().collect();
    let index = (0..reference.len().max(fazo.len()))
        .find(|&index| reference.get(index) != fazo.get(index))
        .unwrap_or_default();
    format!(
        "line {}: reference {:?} fazo {:?}",
        index + 1,
        reference.get(index).unwrap_or(&"<none>"),
        fazo.get(index).unwrap_or(&"<none>")
    )
}

fn timeout_scale(args: &[String]) -> Result<u32, String> {
    let Some(position) = args.iter().position(|arg| arg == "--timeout-scale") else {
        return Ok(1);
    };
    let value = args.get(position + 1).ok_or("--timeout-scale needs a number")?;
    match value.parse::<u32>() {
        Ok(scale) if scale >= 1 => Ok(scale),
        _ => Err(format!("--timeout-scale needs a whole number of at least 1, not {value}")),
    }
}

fn jit_mode(args: &[String]) -> Result<(), String> {
    let Some(position) = args.iter().position(|arg| arg == "--jit") else {
        return Ok(());
    };
    match args.get(position + 1).and_then(|name| differential::JitMode::from_name(name)) {
        Some(mode) => {
            differential::set_jit_mode(mode);
            Ok(())
        }
        None => Err("--jit needs normal, interpreter, baseline or peak".to_string()),
    }
}

fn entry(args: &[String]) -> Result<Run, String> {
    let Some(position) = args.iter().position(|arg| arg == "--entry") else {
        return Ok(match differential::jit_mode() {
            differential::JitMode::Normal => Run::Normal,
            differential::JitMode::Interpreter => Run::Interpreter,
            differential::JitMode::Baseline => Run::Baseline,
            differential::JitMode::Peak => Run::Peak,
        });
    };
    args.get(position + 1)
        .and_then(|name| Run::from_name(name))
        .ok_or_else(|| "--entry needs normal, interpreter, baseline, peak, gc-stress or debug".to_string())
}

fn seed_count(args: &[String]) -> Result<u64, String> {
    let Some(position) = args.iter().position(|arg| arg == "--seeds") else {
        return Ok(DEFAULT_SEEDS);
    };
    let value = args.get(position + 1).ok_or("--seeds needs a number")?;
    value.parse().map_err(|error| format!("--seeds: {error}"))
}

fn scoreboard(seeds: u64, run: Run) -> ExitCode {
    let reference = match reference::shared() {
        Ok(reference) => reference,
        Err(error) => return fail(&error),
    };
    let fazo_exe = fazo_binary();
    if !fazo_exe.is_file() {
        return fail(&format!(
            "no fazo executable at {}; build it with `cargo build --manifest-path crates/fazo_cli/Cargo.toml` or set FAZO_CLI",
            fazo_exe.display()
        ));
    }
    let official_rows = match official_rows(&reference.suite, &reference.lua, &fazo_exe) {
        Ok(rows) => rows,
        Err(error) => return fail(&format!("cannot read the official suite: {error}")),
    };
    let case_rows = match case_rows() {
        Ok(rows) => rows,
        Err(error) => return fail(&format!("cannot read the cases: {error}")),
    };
    let syntax_rows = match syntax_rows() {
        Ok(rows) => rows,
        Err(error) => return fail(&format!("cannot read the syntax cases: {error}")),
    };
    let seed_reports = match [Profile::Full, Profile::Metatables, Profile::Core].map(|profile| seed_report(seeds, profile)) {
        [Ok(full), Ok(metatables), Ok(core)] => [full, metatables, core],
        [Err(error), _, _] | [_, Err(error), _] | [_, _, Err(error)] => return fail(&error),
    };
    let command_rows = match cli::run_all(&fazo_exe) {
        Ok(rows) => rows,
        Err(error) => return fail(&error),
    };
    let standalone = standalone_main(&reference.lua, &fazo_exe, &reference.suite);
    let path = report::update(|board| {
        let tests = &mut board.tests;
        *tests.official.entry(run) = Some(official_suite(&official_rows));
        *tests.cases.entry(run) = Some(verdict_suite(case_rows.iter().map(|row| (row.name.as_str(), &row.fazo))));
        *tests.syntax.entry(run) = Some(verdict_suite(syntax_rows.iter().map(|row| (row.name.as_str(), &row.fazo))));
        *tests.cli.entry(run) = Some(command_suite(&command_rows, &standalone));
        *tests.generated.entry(run) = Some(generated_suite(&seed_reports));
    });
    let path = match path {
        Ok(path) => path,
        Err(error) => return fail(&error),
    };
    print!(
        "{}",
        summary(&official_rows, &case_rows, &syntax_rows, &seed_reports, &command_rows, &standalone)
    );
    println!("report: {}", path.display());
    ExitCode::SUCCESS
}

fn official_suite(rows: &[OfficialRow]) -> Suite {
    let mut suite = Suite::default();
    for row in rows {
        match &row.fazo {
            Column::NotRun => suite.record_skip(),
            Column::Ran(verdict) if verdict.is_pass() => suite.record_pass(),
            Column::Ran(_) => suite.record_fail(row.name.as_str()),
        }
    }
    suite
}

fn verdict_suite<'a>(rows: impl Iterator<Item = (&'a str, &'a Verdict)>) -> Suite {
    let mut suite = Suite::default();
    for (name, verdict) in rows {
        if verdict.is_pass() {
            suite.record_pass();
            continue;
        }
        suite.record_fail(name);
    }
    suite
}

fn command_suite(rows: &[cli::Outcome], standalone: &StandaloneRow) -> Suite {
    let mut suite = Suite::default();
    for row in rows {
        match (row.passed(), row.gap) {
            (true, _) => suite.record_pass(),
            (false, Some(_)) => suite.record_skip(),
            (false, None) => suite.record_fail(row.name),
        }
    }
    if standalone.fazo.is_pass() {
        suite.record_pass();
    } else {
        suite.record_fail(STANDALONE_CASE);
    }
    suite
}

fn generated_suite(reports: &[SeedReport]) -> Suite {
    let mut suite = Suite::default();
    for report in reports {
        suite.pass += report.matched as usize;
        suite.skip += report.rejected as usize;
        for seed in &report.mismatching_seeds {
            suite.record_fail(format!("{}/{seed}", profile_key(report.profile)));
        }
    }
    suite
}

fn fazo_binary() -> PathBuf {
    if let Some(path) = env::var_os("FAZO_CLI") {
        return PathBuf::from(path);
    }
    let target = match env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(CRATE_DIR).join("..").join("..").join("target"),
    };
    target.join("debug").join(format!("fazo{}", env::consts::EXE_SUFFIX))
}

fn standalone_main(lua: &Path, fazo_exe: &Path, suite: &Path) -> StandaloneRow {
    StandaloneRow {
        reference: standalone_verdict(lua, suite),
        fazo: standalone_verdict(fazo_exe, suite),
    }
}

fn standalone_verdict(program: &Path, suite: &Path) -> Verdict {
    match reference::run_process(program, &["main.lua"], &[], &[], suite, reference::scaled(OFFICIAL_DEADLINE)) {
        Ok(captured) if captured.success => Verdict::Pass,
        Ok(captured) => Verdict::Fail(message_of(&first_line(&captured.stderr))),
        Err(error) => Verdict::Fail(error),
    }
}

fn message_of(line: &str) -> String {
    match line.split_once(": ") {
        Some((_, message)) => message.to_string(),
        None => line.to_string(),
    }
}

fn fail(message: &str) -> ExitCode {
    eprintln!("scoreboard: {message}");
    ExitCode::FAILURE
}

fn official_rows(suite: &Path, lua: &Path, fazo_exe: &Path) -> io::Result<Vec<OfficialRow>> {
    let names = official::files(suite)?;
    Ok(names
        .into_iter()
        .map(|name| official_row(name, suite, lua, fazo_exe))
        .collect())
}

fn official_row(name: String, suite: &Path, lua: &Path, fazo_exe: &Path) -> OfficialRow {
    let parses = parses_column(&name, suite);
    let reference = if official::reference_skips(&name) {
        Column::NotRun
    } else {
        Column::Ran(official_verdict(lua, suite, &name, &[]))
    };
    let fazo = match official::out_of_scope_reason(&name) {
        Some(_) => Column::NotRun,
        None => Column::Ran(official_verdict(fazo_exe, suite, &name, &fazo_flags())),
    };
    OfficialRow {
        name,
        reference,
        fazo,
        parses,
    }
}

fn official_verdict(program: &Path, suite: &Path, name: &str, flags: &[&str]) -> Verdict {
    let mut args = flags.to_vec();
    args.extend(["-e", official::PORTABLE_MODE, name]);
    match reference::execute(program, &args, suite, reference::scaled(OFFICIAL_DEADLINE)) {
        Ok(captured) if captured.success => Verdict::Pass,
        Ok(captured) => Verdict::Fail(first_line(&reference::error_message(program, &captured))),
        Err(error) => Verdict::Fail(error),
    }
}

fn fazo_flags() -> Vec<&'static str> {
    differential::jit_mode().flags().to_vec()
}

fn parses_column(name: &str, suite: &Path) -> Verdict {
    let source = match fs::read(suite.join(name)) {
        Ok(source) => source,
        Err(error) => return Verdict::Fail(error.to_string()),
    };
    parse_verdict(name, &official::in_portable_mode(&source))
}

fn parse_verdict(name: &str, source: &[u8]) -> Verdict {
    match fazo::syntax::parse(format!("@{name}").as_bytes(), source) {
        Ok(_) => Verdict::Pass,
        Err(error) => Verdict::Fail(first_line(&error)),
    }
}

fn case_rows() -> io::Result<Vec<CaseRow>> {
    let mut rows = Vec::new();
    for path in lua_files(Path::new(CASES_DIR))? {
        let name = file_name(&path);
        let source = fs::read(&path)?;
        let fazo = check_case(&name, &source);
        let parses = parse_verdict(&name, &source);
        rows.push(CaseRow { name, fazo, parses });
    }
    Ok(rows)
}

fn syntax_rows() -> io::Result<Vec<SyntaxRow>> {
    let mut rows = Vec::new();
    for path in lua_files(&Path::new(CASES_DIR).join(SYNTAX_DIR))? {
        let name = file_name(&path);
        let source = fs::read(&path)?;
        let fazo = match differential::compare(&source) {
            Ok(comparison) => verdict(&comparison),
            Err(error) => Verdict::Fail(error),
        };
        rows.push(SyntaxRow { name, fazo });
    }
    Ok(rows)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn check_case(name: &str, source: &[u8]) -> Verdict {
    match differential::compare(source) {
        Ok(comparison) if comparison.matches() => Verdict::Pass,
        Ok(comparison) => match KNOWN_PUC_BUGS.iter().find(|(case, _)| *case == name) {
            Some((_, reason)) if comparison.fazo.error.is_none() => Verdict::KnownPucBug(reason),
            _ => verdict(&comparison),
        },
        Err(error) => Verdict::Fail(error),
    }
}

fn verdict(comparison: &Comparison) -> Verdict {
    if comparison.matches() {
        return Verdict::Pass;
    }
    Verdict::Fail(describe_mismatch(comparison))
}

fn describe_mismatch(comparison: &Comparison) -> String {
    match &comparison.fazo.error {
        Some(error) => format!("error: {}", first_line(error)),
        None => mismatch_signature(comparison),
    }
}

fn seed_report(seeds: u64, profile: Profile) -> Result<SeedReport, String> {
    let mut report = SeedReport {
        profile,
        seeds,
        matched: 0,
        rejected: 0,
        mismatching_seeds: Vec::new(),
    };
    for seed in 0..seeds {
        let source = generator::program_for(seed, profile);
        let comparison = differential::compare(source.as_bytes())?;
        record_seed(&mut report, seed, &comparison);
    }
    Ok(report)
}

fn record_seed(report: &mut SeedReport, seed: u64, comparison: &Comparison) {
    if comparison.reference.error.is_some() {
        report.rejected += 1;
        return;
    }
    if comparison.matches() {
        report.matched += 1;
        return;
    }
    report.mismatching_seeds.push(seed);
}

fn mismatch_persists(lines: &[String]) -> bool {
    let source = lines.join("\n") + "\n";
    match differential::compare(source.as_bytes()) {
        Ok(comparison) => comparison.reference.error.is_none() && !comparison.matches(),
        Err(_) => false,
    }
}

fn profile_key(profile: Profile) -> &'static str {
    match profile {
        Profile::Full => "full",
        Profile::Metatables => "metatables",
        Profile::Core => "core",
    }
}

fn profile_name(profile: Profile) -> &'static str {
    match profile {
        Profile::Full => "Full",
        Profile::Metatables => "Metatables",
        Profile::Core => "Core",
    }
}

fn summary(
    official_rows: &[OfficialRow],
    case_rows: &[CaseRow],
    syntax_rows: &[SyntaxRow],
    seed_reports: &[SeedReport],
    command_rows: &[cli::Outcome],
    standalone: &StandaloneRow,
) -> String {
    let commands_passing = command_rows.iter().filter(|row| row.passed()).count();
    let command_gaps = command_rows.iter().filter(|row| row.gap.is_some() && !row.passed()).count();
    let commands = command_rows.len();
    let (reference_passing, reference_ran) = tally(official_rows.iter().map(|row| &row.reference));
    let (fazo_passing, fazo_ran) = tally(official_rows.iter().map(|row| &row.fazo));
    let parsing_files = official_rows.iter().filter(|row| row.parses.is_pass()).count();
    let cases_passing = case_rows.iter().filter(|row| row.fazo.is_pass()).count();
    let cases_parsing = case_rows.iter().filter(|row| row.parses.is_pass()).count();
    let syntax_passing = syntax_rows.iter().filter(|row| row.fazo.is_pass()).count();
    let listed = official_rows.len();
    let out_of_scope = listed - fazo_ran;
    let cases = case_rows.len();
    let syntax = syntax_rows.len();
    let generated: String = seed_reports
        .iter()
        .map(|report| {
            format!(
                "generated programs ({}): {} of {} match PUC Lua 5.5.1 ({} rejected by PUC Lua 5.5.1)\n",
                profile_name(report.profile),
                report.matched,
                report.seeds,
                report.rejected
            )
        })
        .collect();
    format!(
        "official files: {listed} listed, {out_of_scope} out of scope\n\
         PUC Lua 5.5.1 passes {reference_passing} of {reference_ran} official files it runs\n\
         fazo passes {fazo_passing} of {fazo_ran} in-scope official files\n\
         official files that parse on fazo: {parsing_files} of {listed}\n\
         hand-written cases: {cases_passing} of {cases} pass on fazo; {cases_parsing} of {cases} parse\n\
         syntax error cases: {syntax_passing} of {syntax} match PUC Lua 5.5.1 exactly\n\
         command-line cases: {commands_passing} of {commands} match PUC Lua 5.5.1 ({command_gaps} known gaps)\n\
         standalone main.lua: PUC Lua 5.5.1 {}; fazo {}\n\
         {generated}",
        standalone.reference.cell(),
        standalone.fazo.cell(),
    )
}

fn tally<'a>(columns: impl Iterator<Item = &'a Column>) -> (usize, usize) {
    columns.fold((0, 0), |(passing, ran), column| match column {
        Column::Ran(verdict) => (passing + usize::from(verdict.is_pass()), ran + 1),
        Column::NotRun => (passing, ran),
    })
}

fn first_line(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    text.lines().next().unwrap_or_default().to_string()
}
