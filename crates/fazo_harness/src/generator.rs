/// Feature sets of generated programs. `Metatables` adds metatables and the string library; `Full` adds `pcall`.
/// `Core` uses neither, since string arithmetic coercion lives in the string metatable in PUC Lua 5.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Full,
    Metatables,
    Core,
}

/// Deterministic Lua programs for differential testing. Expressions have a known type, so divisors are nonzero literals.
/// Programs run without error on PUC Lua 5.5 except inside `pcall`, which only the `Full` profile uses.
pub fn program(seed: u64) -> String {
    program_for(seed, Profile::Full)
}

pub fn program_for(seed: u64, profile: Profile) -> String {
    let mut text = lines_for(seed, profile).join("\n");
    text.push('\n');
    text
}

pub fn lines(seed: u64) -> Vec<String> {
    lines_for(seed, Profile::Full)
}

pub fn lines_for(seed: u64, profile: Profile) -> Vec<String> {
    Generator::new(seed, profile).program()
}

const PREAMBLE: &[&str] = &[
    "local t = {}",
    "local function val(x) if type(x) == \"table\" then return x.v end return x end",
    "local V = {}",
    "V.__add = function(a, b) return val(a) + val(b) end",
    "V.__sub = function(a, b) return val(a) - val(b) end",
    "V.__mul = function(a, b) return val(a) * val(b) end",
    "V.__div = function(a, b) return val(a) / val(b) end",
    "V.__mod = function(a, b) return val(a) % val(b) end",
    "V.__pow = function(a, b) return val(a) ^ val(b) end",
    "V.__idiv = function(a, b) return val(a) // val(b) end",
    "V.__band = function(a, b) return val(a) & val(b) end",
    "V.__bor = function(a, b) return val(a) | val(b) end",
    "V.__bxor = function(a, b) return val(a) ~ val(b) end",
    "V.__shl = function(a, b) return val(a) << val(b) end",
    "V.__shr = function(a, b) return val(a) >> val(b) end",
    "V.__unm = function(a) return -val(a) end",
    "V.__bnot = function(a) return ~val(a) end",
    "V.__len = function(a) return val(a) * 10 end",
    "V.__concat = function(a, b) return \"<\" .. tostring(val(a)) .. \"|\" .. tostring(val(b)) .. \">\" end",
    "V.__eq = function(a, b) return val(a) == val(b) end",
    "V.__lt = function(a, b) return val(a) < val(b) end",
    "V.__le = function(a, b) return val(a) <= val(b) end",
    "V.__call = function(self, a) return val(self) + val(a) end",
    "V.__index = function(_, k) return #tostring(k) end",
    "local vA = setmetatable({v = 5}, V)",
    "local vB = setmetatable({v = -3}, V)",
    "local function mk(start) local c = start return function(d) c = c + d return c end end",
    "local function va(...) return select(\"#\", ...), ... end",
    "local function mr(a) return a, a * 2, a + 1 end",
    "local function tc(n, acc) if n == 0 then return acc end return tc(n - 1, acc + n) end",
    "local w = setmetatable({}, {__newindex = function(tb, k, v) rawset(tb, k, v * 2) end})",
];

const TAIL: &[&str] = &[
    "local vC = setmetatable({v = 1}, V)",
    "print(vC + 1)",
    "setmetatable(vC, nil)",
    "print(pcall(function() return vC + 1 end))",
];

const INT_LITERALS: &[&str] = &[
    "0",
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "100",
    "255",
    "65535",
    "2147483647",
    "4294967296",
    "9007199254740993",
    "9223372036854775807",
];

const NONZERO_LITERALS: &[&str] = &["1", "2", "3", "7", "100", "(-2)", "(-7)"];

const SHIFT_AMOUNTS: &[&str] = &["0", "1", "3", "31", "63", "64", "(-1)"];

const FLOAT_LITERALS: &[&str] = &[
    "0.5",
    "1.5",
    "2.25",
    "3.0",
    "0.1",
    "1e100",
    "1e-7",
    "9007199254740992.0",
    "(-0.0)",
    "(-2.25)",
];

const TABLE_KEYS: &[&str] = &["1", "2", "3", "4", "\"a\"", "\"b\""];

const STRING_LITERALS: &[&str] = &["ab", "x y", "10", "1.5", "0x10", "-7", " 5 ", "1e2"];

const NUMERIC_STRINGS: &[&str] = &["7", "12", "0x1F", "1e2", " 5 "];

const INTEGER_STRINGS: &[&str] = &["7", "12", "0x1F", " 5 "];

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

struct Generator {
    rng: SplitMix64,
    lines: Vec<String>,
    ints: Vec<String>,
    floats: Vec<String>,
    strings: Vec<String>,
    serial: usize,
    profile: Profile,
}

impl Generator {
    fn new(seed: u64, profile: Profile) -> Self {
        Self {
            rng: SplitMix64(seed),
            lines: Vec::new(),
            ints: Vec::new(),
            floats: Vec::new(),
            strings: Vec::new(),
            serial: 0,
            profile,
        }
    }

    fn program(mut self) -> Vec<String> {
        let preamble = PREAMBLE
            .iter()
            .filter(|line| self.profile != Profile::Core || !line.contains("setmetatable"));
        self.lines.extend(preamble.map(|line| line.to_string()));
        let statements = 30 + self.rng.below(30);
        for _ in 0..statements {
            self.statement();
        }
        if self.profile == Profile::Full {
            self.lines.extend(TAIL.iter().map(|line| line.to_string()));
        }
        self.lines
    }

    fn fresh(&mut self, prefix: char) -> String {
        self.serial += 1;
        format!("{prefix}{}", self.serial)
    }

    fn statement(&mut self) {
        match self.rng.below(16) {
            0 => self.declare_int(),
            1 => self.declare_float(),
            2 => self.declare_string(),
            3 => self.table_store(),
            4 => self.print_comparison(),
            5 => self.print_arithmetic(),
            6 if self.profile != Profile::Core => self.metamethod_line(),
            6 => self.print_arithmetic(),
            7 => self.closure_line(),
            8 => self.loop_line(),
            9 if self.profile == Profile::Full => self.protected_error_line(),
            9 => self.print_comparison(),
            10 => self.bitwise_line(),
            11 => self.multiple_results_line(),
            12 => self.coercion_line(),
            13 => self.conversion_line(),
            14 => self.string_line(),
            _ => self.declare_int(),
        }
    }

    fn declare_int(&mut self) {
        let name = self.fresh('i');
        let value = self.int_expr(3);
        self.lines.push(format!("local {name} = {value}"));
        self.lines.push(format!("print({name})"));
        self.ints.push(name);
    }

    fn declare_float(&mut self) {
        let name = self.fresh('f');
        let value = self.float_expr(3);
        self.lines.push(format!("local {name} = {value}"));
        self.lines.push(format!("print({name})"));
        self.floats.push(name);
    }

    fn declare_string(&mut self) {
        let name = self.fresh('s');
        let value = self.str_expr(2);
        self.lines.push(format!("local {name} = {value}"));
        self.lines.push(format!("print({name})"));
        self.strings.push(name);
    }

    fn table_store(&mut self) {
        let key = *self.rng.pick(TABLE_KEYS);
        let value = self.int_expr(2);
        self.lines.push(format!("t[{key}] = {value}"));
        self.lines.push(format!("print(t[{key}] or 0)"));
    }

    fn print_comparison(&mut self) {
        let condition = self.bool_expr();
        self.lines.push(format!("print({condition})"));
    }

    fn print_arithmetic(&mut self) {
        let integer = self.int_expr(2);
        let float = self.float_expr(2);
        self.lines.push(format!("print({integer}, {float})"));
    }

    fn metamethod_line(&mut self) {
        let value = self.int_expr(2);
        let nonzero = *self.rng.pick(NONZERO_LITERALS);
        let line = match self.rng.below(11) {
            0 => format!("print(vA + {value})"),
            1 => format!("print(vB - {value})"),
            2 => format!("print(vA * {value}, vB // {nonzero})"),
            3 => format!("print(vA % {nonzero}, vB / {nonzero})"),
            4 => format!("print(vA == vB, vA < {value}, {value} <= vB)"),
            5 => "print(vA .. \"s\", #vA, -vB)".to_string(),
            6 => format!("print(vA({value}), vB.v, vA.zzz)"),
            7 => format!("print(rawequal(vA, vB), vB & {value}, vA | {value}, ~vB)"),
            8 => "print(vA ^ 2, vB >> 1, vA << 2)".to_string(),
            9 => "print(vA + 0.5, vA ^ 0.5, vB .. vA)".to_string(),
            _ => {
                self.lines.push(format!("w.k = {value}"));
                "print(rawget(w, \"k\"), w.k)".to_string()
            }
        };
        self.lines.push(line);
    }

    fn closure_line(&mut self) {
        let name = self.fresh('c');
        let start = self.int_expr(1);
        let argument = self.int_expr(1);
        self.lines.push(format!("local {name} = mk({start})"));
        self.lines
            .push(format!("print({name}({argument}), {name}(0))"));
    }

    fn loop_line(&mut self) {
        if self.rng.chance(70) {
            self.integer_loop();
        } else {
            self.float_loop();
        }
    }

    fn integer_loop(&mut self) {
        let start = 1 + self.rng.below(3);
        let limit = 3 + self.rng.below(5);
        let step = *self.rng.pick(&["1", "2", "3"]);
        self.ints.push("i".to_string());
        let body = self.int_expr(2);
        self.ints.pop();
        self.lines
            .push(format!("for i = {start}, {limit}, {step} do print(i, {body}) end"));
    }

    fn float_loop(&mut self) {
        self.floats.push("x".to_string());
        let body = self.float_expr(2);
        self.floats.pop();
        self.lines
            .push(format!("for x = 0.5, 2.0, 0.5 do print(x, {body}) end"));
    }

    fn protected_error_line(&mut self) {
        let integer = self.int_expr(1);
        let serial = self.serial;
        let line = match self.rng.below(7) {
            0 => format!("print(pcall(function() return {integer} // 0 end))"),
            1 => format!("print(pcall(function() return {integer} % 0 end))"),
            2 => format!("print(pcall(function() return 1.5 | {integer} end))"),
            3 => "print(pcall(function() return 1 < \"x\" end))".to_string(),
            4 => format!("print(pcall(error, \"boom{serial}\", 0))"),
            5 => "print(pcall(function() error(\"lvl\") end))".to_string(),
            _ => "print(pcall(nil))".to_string(),
        };
        self.lines.push(line);
    }

    fn bitwise_line(&mut self) {
        let left = self.int_expr(2);
        let right = self.int_expr(2);
        let shift = *self.rng.pick(SHIFT_AMOUNTS);
        self.lines.push(format!(
            "print({left} & {right}, {left} | {right}, {left} ~ {right}, {left} << {shift}, {left} >> {shift}, ~({left}))"
        ));
    }

    fn multiple_results_line(&mut self) {
        let line = match self.rng.below(4) {
            0 => format!("print(mr({}))", self.int_expr(2)),
            1 => {
                let number = self.number_expr(1);
                let text = *self.rng.pick(STRING_LITERALS);
                format!("print(va({number}, nil, \"{text}\"))")
            }
            2 => {
                let first = self.int_expr(1);
                let second = self.int_expr(1);
                format!("print(select(-1, {first}, {second}))")
            }
            _ => format!("print(tc({}, 0))", self.rng.below(31)),
        };
        self.lines.push(line);
    }

    fn coercion_line(&mut self) {
        if self.profile == Profile::Core {
            return self.print_arithmetic();
        }
        let digits = *self.rng.pick(NUMERIC_STRINGS);
        let integer = self.int_expr(1);
        self.lines.push(format!(
            "print(\"{digits}\" + {integer}, \"{digits}\" * {integer}, \"{digits}\" // 1)"
        ));
    }

    fn conversion_line(&mut self) {
        let float = self.float_expr(2);
        let integer = self.int_expr(1);
        self.lines.push(format!(
            "print(math.type({float}), math.tointeger({float}), math.type({integer}))"
        ));
    }

    fn string_line(&mut self) {
        if self.strings.is_empty() {
            return self.declare_string();
        }
        let name = self.rng.pick(&self.strings).clone();
        let line = match self.profile {
            Profile::Core => format!("print(#{name}, {name} == \"x\")"),
            Profile::Metatables | Profile::Full => format!("print({name}:upper(), #{name}, {name} == \"x\")"),
        };
        self.lines.push(line);
    }

    fn int_expr(&mut self, depth: u32) -> String {
        if depth == 0 || self.rng.chance(20) {
            return self.int_atom();
        }
        let inner = depth - 1;
        match self.rng.below(9) {
            0..=2 => {
                let op = *self.rng.pick(&["+", "-", "*"]);
                let left = self.int_expr(inner);
                let right = self.int_expr(inner);
                format!("({left} {op} {right})")
            }
            3 => {
                let op = *self.rng.pick(&["//", "%"]);
                let left = self.int_expr(inner);
                let right = *self.rng.pick(NONZERO_LITERALS);
                format!("({left} {op} {right})")
            }
            4 => {
                let op = *self.rng.pick(&["&", "|", "~"]);
                let left = self.int_expr(inner);
                let right = self.int_expr(inner);
                format!("({left} {op} {right})")
            }
            5 => {
                let op = *self.rng.pick(&["<<", ">>"]);
                let left = self.int_expr(inner);
                let right = *self.rng.pick(SHIFT_AMOUNTS);
                format!("({left} {op} {right})")
            }
            6 => {
                let operand = self.int_expr(inner);
                format!("(~({operand}))")
            }
            7 => {
                let operand = self.int_expr(inner);
                format!("(-({operand}))")
            }
            _ => self.int_conversion(inner),
        }
    }

    fn int_conversion(&mut self, depth: u32) -> String {
        if self.profile == Profile::Core || self.rng.chance(50) {
            let operand = self.float_expr(depth);
            return format!("(math.tointeger({operand}) or 0)");
        }
        let digits = *self.rng.pick(INTEGER_STRINGS);
        let operand = self.int_expr(depth);
        format!("(\"{digits}\" + {operand})")
    }

    fn int_atom(&mut self) -> String {
        if !self.ints.is_empty() && self.rng.chance(40) {
            return self.rng.pick(&self.ints).clone();
        }
        match self.rng.below(5) {
            0 => format!("(t[{}] or 0)", self.rng.pick(TABLE_KEYS)),
            1 => (*self.rng.pick(&["math.maxinteger", "math.mininteger"])).to_string(),
            2 if !self.strings.is_empty() => format!("(#{})", self.rng.pick(&self.strings)),
            _ => self.int_literal(),
        }
    }

    fn int_literal(&mut self) -> String {
        let magnitude = *self.rng.pick(INT_LITERALS);
        if self.rng.chance(30) {
            format!("(-{magnitude})")
        } else {
            magnitude.to_string()
        }
    }

    fn float_expr(&mut self, depth: u32) -> String {
        if depth == 0 || self.rng.chance(20) {
            return self.float_atom();
        }
        let inner = depth - 1;
        match self.rng.below(7) {
            0 | 1 => {
                let op = *self.rng.pick(&["+", "-", "*"]);
                let left = self.float_expr(inner);
                let right = self.number_expr(inner);
                format!("({left} {op} {right})")
            }
            2 => {
                let op = *self.rng.pick(&["+", "-", "*"]);
                let left = self.number_expr(inner);
                let right = self.float_expr(inner);
                format!("({left} {op} {right})")
            }
            3 => {
                let left = self.float_expr(inner);
                let right = *self.rng.pick(NONZERO_LITERALS);
                format!("({left} / {right})")
            }
            4 => {
                let op = *self.rng.pick(&["//", "%"]);
                let left = self.float_expr(inner);
                let right = *self.rng.pick(NONZERO_LITERALS);
                format!("({left} {op} {right})")
            }
            5 => {
                let base = self.float_expr(inner);
                format!("({base} ^ 2)")
            }
            _ => {
                let integer = self.int_expr(inner);
                let float = self.float_expr(inner);
                format!("({integer} + {float})")
            }
        }
    }

    fn float_atom(&mut self) -> String {
        if !self.floats.is_empty() && self.rng.chance(40) {
            return self.rng.pick(&self.floats).clone();
        }
        if self.rng.chance(30) {
            return format!("({} + 0.5)", self.int_expr(1));
        }
        (*self.rng.pick(FLOAT_LITERALS)).to_string()
    }

    fn number_expr(&mut self, depth: u32) -> String {
        if self.rng.chance(50) {
            self.int_expr(depth)
        } else {
            self.float_expr(depth)
        }
    }

    fn str_expr(&mut self, depth: u32) -> String {
        if depth == 0 || self.rng.chance(30) {
            return self.str_atom();
        }
        let inner = depth - 1;
        let choice = self.rng.below(5);
        let choice = if self.profile == Profile::Core && choice >= 2 { 0 } else { choice };
        match choice {
            0 => {
                let left = self.str_expr(inner);
                let right = self.str_expr(inner);
                format!("({left} .. {right})")
            }
            1 => {
                let number = self.number_expr(inner);
                let text = self.str_expr(inner);
                format!("({number} .. {text})")
            }
            2 => {
                let text = *self.rng.pick(STRING_LITERALS);
                format!("(\"{text}\"):rep({})", self.rng.below(4))
            }
            3 => {
                let integer = self.int_expr(inner);
                let float = self.float_expr(inner);
                format!("string.format(\"%d|%5.2f\", {integer}, {float})")
            }
            _ => {
                let text = self.str_expr(inner);
                format!("string.sub({text}, 1, 3)")
            }
        }
    }

    fn str_atom(&mut self) -> String {
        match self.rng.below(3) {
            0 => format!("\"{}\"", self.rng.pick(STRING_LITERALS)),
            1 => format!("tostring({})", self.number_expr(1)),
            _ => format!("\"{}\"", self.rng.pick(STRING_LITERALS)),
        }
    }

    fn bool_expr(&mut self) -> String {
        match self.rng.below(5) {
            0 => format!("({} < {})", self.number_expr(1), self.number_expr(1)),
            1 => format!("({} == {})", self.number_expr(1), self.number_expr(1)),
            2 => format!("({} <= {})", self.number_expr(1), self.number_expr(1)),
            3 => format!("({} ~= {})", self.number_expr(1), self.number_expr(1)),
            _ => format!("({} < {})", self.str_expr(1), self.str_expr(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{lines, program};
    use crate::differential::{CHUNK_NAME, run_reference};

    #[test]
    fn programs_are_deterministic_per_seed() {
        assert_eq!(program(7), program(7));
        assert_ne!(program(7), program(8));
        assert_eq!(lines(7).len(), program(7).lines().count());
    }

    #[test]
    fn generated_programs_run_clean_on_the_reference() {
        for seed in 0..64 {
            let outcome = run_reference(program(seed).as_bytes(), CHUNK_NAME)
                .expect("the reference is cached; see reference::shared for how to populate it");
            assert_eq!(
                outcome.error, None,
                "seed {seed} errors on the reference:\n{}",
                program(seed)
            );
        }
    }
}
