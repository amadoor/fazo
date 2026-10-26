use std::{
    fs, io,
    path::{Path, PathBuf},
};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");
const LUA_VERSION: &str = "5.5.1";

#[derive(Debug, Serialize, Deserialize)]
pub struct Scoreboard {
    pub lua: String,
    #[serde(default)]
    pub tests: Tests,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytecode: Option<Bytecode>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub perf: IndexMap<String, Workload>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub operations: IndexMap<String, Operation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_cost: Option<CompileCost>,
}

impl Default for Scoreboard {
    fn default() -> Self {
        Self {
            lua: LUA_VERSION.to_owned(),
            tests: Tests::default(),
            bytecode: None,
            perf: IndexMap::new(),
            operations: IndexMap::new(),
            compile_cost: None,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Tests {
    #[serde(default)]
    pub official: Tiers,
    #[serde(default)]
    pub cases: Tiers,
    #[serde(default)]
    pub syntax: Tiers,
    #[serde(default)]
    pub cli: Tiers,
    #[serde(default)]
    pub generated: Tiers,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Tiers {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<Suite>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpreter: Option<Suite>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Suite>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak: Option<Suite>,
    #[serde(default, rename = "gc-stress", skip_serializing_if = "Option::is_none")]
    pub gc_stress: Option<Suite>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<Suite>,
}

impl Tiers {
    pub fn entry(&mut self, run: Run) -> &mut Option<Suite> {
        match run {
            Run::Normal => &mut self.normal,
            Run::Interpreter => &mut self.interpreter,
            Run::Baseline => &mut self.baseline,
            Run::Peak => &mut self.peak,
            Run::GcStress => &mut self.gc_stress,
            Run::Debug => &mut self.debug,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    Normal,
    Interpreter,
    Baseline,
    Peak,
    GcStress,
    Debug,
}

impl Run {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "normal" => Some(Self::Normal),
            "interpreter" => Some(Self::Interpreter),
            "baseline" => Some(Self::Baseline),
            "peak" => Some(Self::Peak),
            "gc-stress" => Some(Self::GcStress),
            "debug" => Some(Self::Debug),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Suite {
    pub pass: usize,
    pub fail: usize,
    pub skip: usize,
    pub failures: Vec<String>,
}

impl Suite {
    pub fn record_pass(&mut self) {
        self.pass += 1;
    }

    pub fn record_skip(&mut self) {
        self.skip += 1;
    }

    pub fn record_fail(&mut self, name: impl Into<String>) {
        self.fail += 1;
        self.failures.push(name.into());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bytecode {
    pub functions: usize,
    pub identical: usize,
    pub fazo_instructions: u64,
    pub puc_instructions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workload {
    pub puc: i64,
    pub fazo: i64,
    pub baseline: i64,
    pub peak: i64,
    pub interpreter: i64,
    pub luajit: Option<i64>,
    pub luajit_interpreter: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub luajit_missing: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub puc: f64,
    pub fazo: f64,
    #[serde(default)]
    pub baseline: f64,
    #[serde(default)]
    pub peak: f64,
    #[serde(default)]
    pub luajit: f64,
    pub budget: f64,
}

/// Peak compile Ir per bytecode for one function of `statements` global updates run 200 times, and LuaJIT's on the same code.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompileCost {
    pub statements: u32,
    pub peak_per_bytecode: f64,
    pub luajit_per_bytecode: f64,
    pub peak_per_statement: f64,
    pub luajit_per_statement: f64,
}

pub fn path() -> PathBuf {
    Path::new(CRATE_DIR).join("..").join("fazo").join("scoreboard.json")
}

pub fn update(change: impl FnOnce(&mut Scoreboard)) -> Result<PathBuf, String> {
    let path = path();
    let existing = match fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("reading {}: {error}", path.display())),
    };
    let text = updated(existing.as_deref(), change).map_err(|error| format!("{}: {error}", path.display()))?;
    fs::write(&path, text).map_err(|error| format!("writing {}: {error}", path.display()))?;
    Ok(path)
}

pub fn updated(existing: Option<&str>, change: impl FnOnce(&mut Scoreboard)) -> Result<String, String> {
    let mut board = match existing {
        Some(text) => serde_json::from_str(text).map_err(|error| format!("reading the scoreboard: {error}"))?,
        None => Scoreboard::default(),
    };
    change(&mut board);
    board.lua = LUA_VERSION.to_owned();

    let mut text = serde_json::to_string_pretty(&board).map_err(|error| format!("writing the scoreboard: {error}"))?;
    text.push('\n');
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::{Bytecode, Operation, Run, Suite, Workload, updated};

    const EXPECTED: &str = r#"{
  "lua": "5.5.1",
  "tests": {
    "official": {
      "normal": {
        "pass": 31,
        "fail": 0,
        "skip": 2,
        "failures": []
      },
      "interpreter": {
        "pass": 30,
        "fail": 1,
        "skip": 2,
        "failures": [
          "coroutine.lua"
        ]
      }
    },
    "cases": {
      "normal": {
        "pass": 120,
        "fail": 0,
        "skip": 0,
        "failures": []
      }
    },
    "syntax": {},
    "cli": {},
    "generated": {}
  },
  "bytecode": {
    "functions": 2729,
    "identical": 2729,
    "fazo_instructions": 122699,
    "puc_instructions": 122699
  },
  "perf": {
    "fib": {
      "puc": 161182880,
      "fazo": 119494966,
      "baseline": 132675213,
      "peak": 239120413,
      "interpreter": 177381266,
      "luajit": 20412555,
      "luajit_interpreter": 98120437
    },
    "garbage collector (generational)": {
      "puc": 1000,
      "fazo": 900,
      "baseline": 950,
      "peak": 800,
      "interpreter": 1100,
      "luajit": null,
      "luajit_interpreter": null,
      "luajit_missing": "no generational collector"
    }
  },
  "operations": {
    "call0": {
      "puc": 157.0,
      "fazo": 158.0,
      "baseline": 21.0,
      "peak": 12.5,
      "luajit": 0.0,
      "budget": 161.0
    }
  }
}
"#;

    #[test]
    fn each_task_replaces_only_its_own_entries_and_keys_keep_their_order() {
        let first = updated(None, |board| {
            board.perf.insert(
                "fib".to_owned(),
                Workload {
                    puc: 161_182_880,
                    fazo: 119_494_966,
                    baseline: 132_675_213,
                    peak: 239_120_413,
                    interpreter: 177_381_266,
                    luajit: Some(20_412_555),
                    luajit_interpreter: Some(98_120_437),
                    luajit_missing: None,
                },
            );
            board.perf.insert(
                "garbage collector (generational)".to_owned(),
                Workload {
                    puc: 1000,
                    fazo: 900,
                    baseline: 950,
                    peak: 800,
                    interpreter: 1100,
                    luajit: None,
                    luajit_interpreter: None,
                    luajit_missing: Some("no generational collector".to_owned()),
                },
            );
            board.operations.insert(
                "call0".to_owned(),
                Operation { puc: 157.0, fazo: 158.0, baseline: 21.0, peak: 12.5, luajit: 0.0, budget: 161.0 },
            );
            *board.tests.official.entry(Run::Interpreter) =
                Some(Suite { pass: 30, fail: 1, skip: 2, failures: vec!["coroutine.lua".to_owned()] });
            *board.tests.official.entry(Run::Normal) = Some(Suite { pass: 1, fail: 1, skip: 0, failures: vec!["stale.lua".to_owned()] });
        })
        .expect("a fresh scoreboard writes");
        let second = updated(Some(&first), |board| {
            board.bytecode = Some(Bytecode { functions: 2729, identical: 2729, fazo_instructions: 122_699, puc_instructions: 122_699 });
            *board.tests.official.entry(Run::Normal) = Some(Suite { pass: 31, fail: 0, skip: 2, failures: Vec::new() });
            *board.tests.cases.entry(Run::Normal) = Some(Suite { pass: 120, ..Suite::default() });
        })
        .expect("the scoreboard reads back");
        assert_eq!(second, EXPECTED);
    }

    #[test]
    fn a_scoreboard_from_before_the_luajit_columns_reads() {
        let old = r#"{"lua": "5.5.1", "perf": {"fib": {"puc": 1, "fazo": 2, "baseline": 3, "peak": 4, "interpreter": 5}},
            "operations": {"call0": {"puc": 1.0, "fazo": 2.0, "budget": 3.0}}}"#;
        let text = updated(Some(old), |_| {}).expect("the older scoreboard reads");
        assert!(text.contains("\"luajit\": null"), "{text}");
        assert!(!text.contains("luajit_missing"), "{text}");
    }

    #[test]
    fn every_run_name_has_an_entry() {
        for name in ["normal", "interpreter", "baseline", "peak", "gc-stress", "debug"] {
            assert!(Run::from_name(name).is_some(), "{name}");
        }
        assert_eq!(Run::from_name("fast"), None);
    }
}
