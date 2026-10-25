use std::fmt::Write as _;
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::Path,
    time::Duration,
};

use fazo::{Options, Vm};

use crate::{
    generator,
    official,
    perf,
    reference::{self, Reference, WorkDir},
    report,
};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const DEADLINE: Duration = Duration::from_secs(120);
const GENERATED_PROGRAMS: u64 = 50;
const TOP_ROWS: usize = 15;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Instruction {
    name: String,
    operands: String,
}

#[derive(Clone, Debug)]
struct Function {
    line_defined: u32,
    last_line: u32,
    instructions: Vec<Instruction>,
}

#[derive(Clone, Debug)]
struct Difference {
    pc: usize,
    fazo: String,
    puc: String,
}

#[derive(Clone, Debug)]
struct FunctionReport {
    label: String,
    line_defined: u32,
    last_line: u32,
    fazo_count: usize,
    puc_count: usize,
    first_difference: Option<Difference>,
}

impl FunctionReport {
    fn excess(&self) -> i64 {
        self.fazo_count as i64 - self.puc_count as i64
    }

    fn identical(&self) -> bool {
        self.first_difference.is_none()
    }
}

#[derive(Default)]
struct FileReport {
    functions: Vec<FunctionReport>,
    unmatched_fazo: usize,
    unmatched_puc: usize,
}

#[derive(Default)]
struct Totals {
    files: usize,
    unmatched_fazo: usize,
    unmatched_puc: usize,
    functions: Vec<FunctionReport>,
}

struct Source {
    label: String,
    name: String,
    bytes: Vec<u8>,
}

pub fn run() -> Result<String, String> {
    let reference = reference::ensure()?;
    let work = WorkDir::create()?;
    let mut totals = Totals::default();
    let mut skipped = Vec::new();
    for (index, source) in sources(&reference)?.iter().enumerate() {
        match compare(&reference, work.path(), index, source) {
            Ok(file) => record(&mut totals, file),
            Err(reason) => skipped.push(format!("{}: {reason}", source.label)),
        }
    }
    totals.functions.sort_by(|a, b| b.excess().cmp(&a.excess()).then(a.label.cmp(&b.label)).then(a.line_defined.cmp(&b.line_defined)));
    let figures = totals_of(&totals);
    let path = report::update(|board| {
        board.bytecode = Some(report::Bytecode {
            functions: figures.functions,
            identical: figures.identical,
            fazo_instructions: figures.fazo,
            puc_instructions: figures.puc,
        });
    })?;
    let mut text = summary(&totals, &skipped);
    let _ = writeln!(text, "\nreport: {}", path.display());
    Ok(text)
}

fn sources(reference: &Reference) -> Result<Vec<Source>, String> {
    let mut sources = perf_sources()?;
    sources.extend(official_sources(&reference.suite)?);
    sources.extend(generated_sources());
    Ok(sources)
}

fn perf_sources() -> Result<Vec<Source>, String> {
    let dir = Path::new(CRATE_DIR).join("perf");
    let mut scripts: Vec<&str> = perf::WORKLOADS.iter().map(|workload| workload.script).collect();
    scripts.sort_unstable();
    scripts.dedup();
    scripts
        .into_iter()
        .map(|name| read_source(&dir.join(name), format!("perf/{name}"), name))
        .collect()
}

fn official_sources(suite: &Path) -> Result<Vec<Source>, String> {
    let names = official::files(suite).map_err(|error| format!("reading {}: {error}", suite.display()))?;
    names
        .into_iter()
        .map(|name| read_source(&suite.join(&name), format!("official/{name}"), &name))
        .collect()
}

fn generated_sources() -> Vec<Source> {
    (0..GENERATED_PROGRAMS)
        .map(|seed| Source {
            label: format!("generated/seed-{seed}"),
            name: format!("generated-{seed}.lua"),
            bytes: generator::program(seed).into_bytes(),
        })
        .collect()
}

fn read_source(path: &Path, label: String, name: &str) -> Result<Source, String> {
    let bytes = fs::read(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    Ok(Source {
        label,
        name: name.to_owned(),
        bytes,
    })
}

fn compare(reference: &Reference, work: &Path, index: usize, source: &Source) -> Result<FileReport, String> {
    let file = format!("{index}-{}", source.name);
    fs::write(work.join(&file), &source.bytes).map_err(|error| format!("writing {file}: {error}"))?;
    let puc = puc_listing(reference, work, &file)?;
    let fazo = fazo_listing(&source.bytes, &format!("@{file}"))?;
    Ok(match_functions(&source.label, &parse_listing(&fazo), &parse_listing(&puc)))
}

fn puc_listing(reference: &Reference, work: &Path, file: &str) -> Result<String, String> {
    let captured = reference::run_process(&reference.luac, &["-p", "-l", file], &[], &[], work, DEADLINE)?;
    if !captured.success {
        return Err(format!("luac: {}", String::from_utf8_lossy(&captured.stderr).trim_end()));
    }
    Ok(String::from_utf8_lossy(&captured.stdout).into_owned())
}

fn fazo_listing(source: &[u8], chunkname: &str) -> Result<String, String> {
    let mut vm = Vm::new(Options::default());
    let function = vm
        .load(chunkname.as_bytes(), &without_shebang(source))
        .map_err(|error| format!("fazo: {}", String::from_utf8_lossy(&vm.error_message(&error))))?;
    let text = vm.listing(function, false).ok_or("fazo: the chunk is not a Lua function")?;
    Ok(String::from_utf8_lossy(&text).into_owned())
}

fn without_shebang(source: &[u8]) -> Vec<u8> {
    if source.first() != Some(&b'#') {
        return source.to_vec();
    }
    let newline = source.iter().position(|&byte| byte == b'\n').unwrap_or(source.len());
    source[newline..].to_vec()
}

fn parse_listing(text: &str) -> Vec<Function> {
    let mut functions: Vec<Function> = Vec::new();
    for line in text.lines() {
        if let Some((line_defined, last_line)) = parse_header(line) {
            functions.push(Function {
                line_defined,
                last_line,
                instructions: Vec::new(),
            });
            continue;
        }
        let Some(instruction) = parse_instruction(line) else {
            continue;
        };
        if let Some(function) = functions.last_mut() {
            function.instructions.push(instruction);
        }
    }
    functions
}

fn parse_header(line: &str) -> Option<(u32, u32)> {
    if !(line.starts_with("main <") || line.starts_with("function <")) {
        return None;
    }
    let inside = line.split_once('<')?.1.split_once('>')?.0;
    let (_, lines) = inside.rsplit_once(':')?;
    let (start, end) = lines.split_once(',')?;
    Some((start.parse().ok()?, end.parse().ok()?))
}

fn parse_instruction(line: &str) -> Option<Instruction> {
    let mut fields = line.strip_prefix('\t')?.splitn(4, '\t');
    fields.next()?.parse::<u32>().ok()?;
    fields.next()?;
    let name = fields.next()?.trim_end().to_owned();
    if name == "RETURN0" {
        return Some(Instruction { name, operands: String::new() });
    }
    let operands = fields.next().unwrap_or("").split("\t; ").next().unwrap_or("").trim_end().to_owned();
    Some(Instruction { name, operands })
}

fn match_functions(label: &str, fazo: &[Function], puc: &[Function]) -> FileReport {
    let mut available: BTreeMap<(u32, u32), VecDeque<Function>> = BTreeMap::new();
    for function in puc {
        available
            .entry((function.line_defined, function.last_line))
            .or_default()
            .push_back(function.clone());
    }
    let mut report = FileReport::default();
    for function in fazo {
        let key = (function.line_defined, function.last_line);
        match available.get_mut(&key).and_then(VecDeque::pop_front) {
            Some(partner) => report.functions.push(compare_functions(label, function, &partner)),
            None => report.unmatched_fazo += 1,
        }
    }
    report.unmatched_puc = available.values().map(VecDeque::len).sum();
    report
}

fn compare_functions(label: &str, fazo: &Function, puc: &Function) -> FunctionReport {
    let fazo_instructions = resolved(&fazo.instructions);
    let puc_instructions = resolved(&puc.instructions);
    FunctionReport {
        label: label.to_owned(),
        line_defined: fazo.line_defined,
        last_line: fazo.last_line,
        fazo_count: fazo_instructions.len(),
        puc_count: puc_instructions.len(),
        first_difference: first_difference(&fazo_instructions, &puc_instructions),
    }
}

fn resolved(instructions: &[Instruction]) -> Vec<Instruction> {
    instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| Instruction {
            name: instruction.name.clone(),
            operands: resolved_operands(instruction, index),
        })
        .collect()
}

fn resolved_operands(instruction: &Instruction, index: usize) -> String {
    let Some(target) = jump_target(instruction, index) else {
        return instruction.operands.clone();
    };
    let first = instruction.operands.split_whitespace().next().unwrap_or("");
    match instruction.name.as_str() {
        "JMP" => format!("->{target}"),
        _ => format!("{first} ->{target}"),
    }
}

fn jump_target(instruction: &Instruction, index: usize) -> Option<i64> {
    let numbers: Vec<i64> = instruction
        .operands
        .split_whitespace()
        .filter_map(|word| word.parse().ok())
        .collect();
    let index = index as i64;
    match (instruction.name.as_str(), numbers.as_slice()) {
        ("JMP", [offset]) => Some(index + 1 + offset),
        ("FORLOOP" | "TFORLOOP", [_, jump]) => Some(index + 1 - jump),
        ("FORPREP", [_, jump]) => Some(index + jump + 2),
        ("TFORPREP", [_, jump]) => Some(index + jump + 1),
        _ => None,
    }
}

fn first_difference(fazo: &[Instruction], puc: &[Instruction]) -> Option<Difference> {
    let length = fazo.len().max(puc.len());
    let index = (0..length).find(|&index| fazo.get(index) != puc.get(index))?;
    Some(Difference {
        pc: index + 1,
        fazo: describe(fazo.get(index)),
        puc: describe(puc.get(index)),
    })
}

fn describe(instruction: Option<&Instruction>) -> String {
    match instruction {
        Some(instruction) if instruction.operands.is_empty() => instruction.name.clone(),
        Some(instruction) => format!("{} {}", instruction.name, instruction.operands),
        None => "(none)".to_owned(),
    }
}

fn record(totals: &mut Totals, file: FileReport) {
    totals.files += 1;
    totals.unmatched_fazo += file.unmatched_fazo;
    totals.unmatched_puc += file.unmatched_puc;
    totals.functions.extend(file.functions);
}

fn totals_of(totals: &Totals) -> Figures {
    let mut figures = Figures::default();
    for function in &totals.functions {
        figures.functions += 1;
        figures.identical += usize::from(function.identical());
        figures.fazo += function.fazo_count as u64;
        figures.puc += function.puc_count as u64;
    }
    figures
}

#[derive(Default)]
struct Figures {
    functions: usize,
    identical: usize,
    fazo: u64,
    puc: u64,
}

fn function_cells(function: &FunctionReport) -> [String; 6] {
    let difference = function.first_difference.as_ref().map_or_else(String::new, |difference| {
        format!("pc {}: fazo `{}`, PUC `{}`", difference.pc, difference.fazo, difference.puc)
    });
    [
        function.label.clone(),
        format!("{}-{}", function.line_defined, function.last_line),
        function.fazo_count.to_string(),
        function.puc_count.to_string(),
        format!("{:+}", function.excess()),
        difference,
    ]
}

fn summary(totals: &Totals, skipped: &[String]) -> String {
    let figures = totals_of(totals);
    let mut text = format!(
        "bytecode: {} files, {} skipped, {} functions matched, {} identical\n\
         instructions: fazo {}, PUC {}, excess {:+}\n\
         unmatched functions: fazo {}, PUC {}\n\n",
        totals.files,
        skipped.len(),
        figures.functions,
        figures.identical,
        figures.fazo,
        figures.puc,
        figures.fazo as i64 - figures.puc as i64,
        totals.unmatched_fazo,
        totals.unmatched_puc,
    );
    let _ = writeln!(text, "top {TOP_ROWS} functions by excess:");
    text.push_str("| source | lines | fazo | PUC | excess | first difference |\n|---|---|---|---|---|---|\n");
    for function in totals.functions.iter().filter(|function| !function.identical()).take(TOP_ROWS) {
        let cells = function_cells(function);
        let _ = writeln!(text, "| {} |", cells.join(" | "));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::{Instruction, parse_listing};

    #[test]
    fn a_listing_splits_into_functions_and_instructions() {
        let text = "\nmain <t.lua:0,0> (2 instructions)\n\
                    0 params, 2 slots, 0 upvalues, 0 locals, 0 constants, 1 function\n\
                    \t1\t[1]\tLOADI     \t0 1\n\
                    \t2\t[-]\tRETURN0   \t\n\
                    \nfunction <t.lua:1,1> (1 instruction)\n\
                    \t1\t[1]\tADDK      \t0 0 1\t; 2\n";
        let functions = parse_listing(text);
        assert_eq!(functions.len(), 2);
        assert_eq!((functions[0].line_defined, functions[0].last_line), (0, 0));
        assert_eq!(
            functions[0].instructions,
            vec![
                Instruction { name: "LOADI".to_owned(), operands: "0 1".to_owned() },
                Instruction { name: "RETURN0".to_owned(), operands: String::new() },
            ]
        );
        assert_eq!(functions[1].instructions[0].operands, "0 0 1");
    }
}
