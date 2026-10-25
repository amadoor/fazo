use crate::{Vm, opcode::Opcode};

const PAIR_COUNT: usize = Opcode::COUNT * Opcode::COUNT;

pub(crate) struct OpcodeCounts {
    single: [u64; Opcode::COUNT],
    pairs: Vec<u64>,
    previous: Option<Opcode>,
}

impl Default for OpcodeCounts {
    fn default() -> Self {
        Self {
            single: [0; Opcode::COUNT],
            pairs: vec![0; PAIR_COUNT],
            previous: None,
        }
    }
}

impl OpcodeCounts {
    pub(crate) fn record(&mut self, opcode: Opcode) {
        self.single[opcode as usize] += 1;
        if let Some(previous) = self.previous {
            self.pairs[previous as usize * Opcode::COUNT + opcode as usize] += 1;
        }
        self.previous = Some(opcode);
    }

    fn report(&self) -> String {
        let mut text = String::from("opcodes:\n");
        for opcode in sorted(self.single_counts()) {
            text.push_str(&format!("{:>14}  {}\n", opcode.1, opcode.0.name()));
        }
        text.push_str("pairs:\n");
        for (count, first, second) in self.sorted_pairs() {
            text.push_str(&format!("{count:>14}  {} {}\n", first.name(), second.name()));
        }
        text
    }

    fn single_counts(&self) -> Vec<(Opcode, u64)> {
        Opcode::ALL
            .iter()
            .map(|&opcode| (opcode, self.single[opcode as usize]))
            .filter(|&(_, count)| count > 0)
            .collect()
    }

    fn sorted_pairs(&self) -> Vec<(u64, Opcode, Opcode)> {
        let mut pairs: Vec<(u64, Opcode, Opcode)> = self
            .pairs
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(index, &count)| (count, Opcode::ALL[index / Opcode::COUNT], Opcode::ALL[index % Opcode::COUNT]))
            .collect();
        pairs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        pairs
    }
}

fn sorted(mut counts: Vec<(Opcode, u64)>) -> Vec<(Opcode, u64)> {
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
}

impl Vm {
    /// Most frequent first; only builds with the `opcode-counts` feature keep counts.
    pub fn opcode_counts_report(&self) -> String {
        self.opcode_counts.report()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Options, Vm};

    #[test]
    fn a_run_counts_each_opcode_and_the_pairs_between_them() {
        let mut vm = Vm::new(Options::default());
        vm.run_source(b"=t", b"local n = 1 local x = n + 2").expect("the chunk runs");
        let report = vm.opcode_counts_report();
        assert!(report.contains("ADDI"), "{report}");
        assert!(report.contains("pairs:"), "{report}");
    }
}
