use std::fmt::{self, Write};

use super::ir::{BlockId, Closure, Const, Edge, Func, HelperKind, Op, Rep, StateId, Term, Type, Val};

const KIND_NAMES: [&str; 11] = ["nil", "bool", "int", "float", "str", "table", "closure", "native", "thread", "userdata", "light"];

impl fmt::Display for Type {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Type::ANY {
            return formatter.write_str("any");
        }
        if *self == Type::NUM {
            return formatter.write_str("num");
        }
        if self.is_none() {
            return formatter.write_str("none");
        }
        let names: Vec<&str> = KIND_NAMES.iter().enumerate().filter(|&(tag, _)| !self.intersect(Type::of_tag(tag as u8)).is_none()).map(|(_, name)| *name).collect();
        formatter.write_str(&names.join("|"))
    }
}

impl fmt::Display for Rep {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Rep::None => "-",
            Rep::Boxed => "boxed",
            Rep::Int => "i64",
            Rep::Float => "f64",
            Rep::Bool => "b",
            Rep::Payload => "ref",
            Rep::Word => "word",
        };
        formatter.write_str(name)
    }
}

impl fmt::Display for Const {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Const::Nil => formatter.write_str("nil"),
            Const::Bool(flag) => write!(formatter, "{flag}"),
            Const::Int(integer) => write!(formatter, "{integer}"),
            Const::Float(float) => write!(formatter, "{float:?}"),
            Const::Gc { tag, payload } => write!(formatter, "{}#{payload:x}", KIND_NAMES.get(usize::from(*tag)).copied().unwrap_or("?")),
        }
    }
}

impl fmt::Display for Func {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut text = String::new();
        for block in self.reverse_postorder() {
            self.write_block(&mut text, block)?;
        }
        if !self.states.is_empty() {
            text.push_str("states:\n");
        }
        for index in 0..self.states.len() {
            self.write_state(&mut text, StateId(index as u32))?;
        }
        formatter.write_str(&text)
    }
}

impl Func {
    fn write_block(&self, text: &mut String, block: BlockId) -> fmt::Result {
        let data = self.block(block);
        write!(text, "block{}", block.0)?;
        if !data.params.is_empty() {
            let params: Vec<String> = data.params.iter().map(|&param| self.typed(param)).collect();
            write!(text, "({})", params.join(", "))?;
        }
        if self.entry == Some(block) {
            text.push_str(" entry");
        }
        if let Some((pc, _)) = self.osr.iter().find(|&&(_, osr)| osr == block) {
            write!(text, " osr@{pc}")?;
        }
        if data.cold {
            text.push_str(" cold");
        }
        text.push_str(":\n");
        for &value in &data.insts {
            self.write_inst(text, value)?;
        }
        text.push_str("  ");
        self.write_term(text, &data.term)?;
        text.push('\n');
        Ok(())
    }

    fn typed(&self, value: Val) -> String {
        let inst = self.inst(value);
        format!("v{}:{}/{}", value.0, inst.ty, inst.rep)
    }

    fn write_inst(&self, text: &mut String, value: Val) -> fmt::Result {
        let inst = self.inst(value);
        text.push_str("  ");
        if inst.rep != Rep::None {
            write!(text, "{} = ", self.typed(value))?;
        }
        text.push_str(&describe(&inst.op));
        text.push('\n');
        Ok(())
    }

    fn write_term(&self, text: &mut String, term: &Term) -> fmt::Result {
        match term {
            Term::Open => text.push_str("open"),
            Term::Jump(edge) => write!(text, "jump {}", edge_text(edge))?,
            Term::Branch { cond, then, other } => write!(text, "branch v{} {} {}", cond.0, edge_text(then), edge_text(other))?,
            Term::Switch { code, cases } => {
                let cases: Vec<String> = cases.iter().map(|(pc, edge)| format!("{pc}:{}", edge_text(edge))).collect();
                write!(text, "switch v{} [{}] else exit", code.0, cases.join(" "))?;
            }
            Term::Return { pc, first, count, close } => write!(text, "return @{pc} r{first} x{count}{}", if *close { " close" } else { "" })?,
            Term::TailCall { pc, a, b, k } => write!(text, "tailcall @{pc} r{a} b{b}{}", if *k { " close" } else { "" })?,
            Term::Deopt { state } => write!(text, "deopt s{}", state.0)?,
        }
        Ok(())
    }

    fn write_state(&self, text: &mut String, state: StateId) -> fmt::Result {
        let data = self.state(state);
        write!(text, "  s{} pc {} site {}", state.0, data.pc, data.site)?;
        for frame in &data.frames {
            write!(text, " frame(base {} caller {} wanted {})", frame.base, frame.caller_pc, frame.wanted)?;
        }
        let slots: Vec<String> = data.slots.iter().map(|(slot, value)| format!("r{slot}=v{}", value.0)).collect();
        let objects: Vec<String> = data.objects.iter().map(virtual_text).collect();
        let lazy = if data.lazy { " lazy" } else { "" };
        writeln!(text, " [{}]{lazy} {}", slots.join(" "), objects.join(" "))
    }
}

fn edge_text(edge: &Edge) -> String {
    let args: Vec<String> = edge.args.iter().map(|arg| format!("v{}", arg.0)).collect();
    if args.is_empty() {
        return format!("block{}", edge.block.0);
    }
    format!("block{}({})", edge.block.0, args.join(", "))
}

fn closure_text(closure: Closure) -> String {
    match closure {
        Closure::Root => "root".to_owned(),
        Closure::Value(value) => format!("v{}", value.0),
    }
}

pub(crate) fn virtual_text(object: &super::ir::Virtual) -> String {
    let fields: Vec<String> = object.fields.iter().map(|field| format!("v{}", field.0)).collect();
    let meta = object.meta.map_or(String::new(), |meta| format!(" meta v{}", meta.0));
    let original = object.original.map_or(String::new(), |(flag, original)| format!(" else v{} unless v{}", original.0, flag.0));
    format!("v{} = {{{:x} [{}]{meta}{original}}}", object.value.0, object.shape, fields.join(", "))
}

pub(crate) fn describe(op: &Op) -> String {
    let v = |value: &Val| format!("v{}", value.0);
    match op {
        Op::Param => "param".to_owned(),
        Op::Const(constant) => format!("const {constant}"),
        Op::LoadReg { slot } => format!("load.reg r{slot}"),
        Op::StoreReg { slot, value } => format!("store.reg r{slot} {}", v(value)),
        Op::LoadUpval { closure, index } => format!("load.upval {} u{index}", closure_text(*closure)),
        Op::StoreUpval { closure, index, value } => format!("store.upval {} u{index} {}", closure_text(*closure), v(value)),
        Op::Unary(op, value) => format!("{} {}", format!("{op:?}").to_lowercase(), v(value)),
        Op::Binary(op, left, right) => format!("{} {}, {}", format!("{op:?}").to_lowercase(), v(left), v(right)),
        Op::Pure(function, args) => {
            let args: Vec<String> = args.iter().map(v).collect();
            format!("pure.{} {}", format!("{function:?}").to_lowercase(), args.join(", "))
        }
        Op::EqConst(value, constant) => format!("eq.const {} {constant}", v(value)),
        Op::Arith { op, left, right, state } => format!("arith.{} {}, {} s{}", format!("{op:?}").to_lowercase(), v(left), v(right), state.0),
        Op::Compare { cc, left, right, state } => format!("compare.{} {}, {} s{}", format!("{cc:?}").to_lowercase(), v(left), v(right), state.0),
        Op::Select { cond, yes, no } => format!("select {}, {}, {}", v(cond), v(yes), v(no)),
        Op::Len { value, state } => format!("len {} s{}", v(value), state.0),
        Op::ShapeIs { table, shape } => format!("shape.is {} {shape:x}", v(table)),
        Op::LoadSlot { table, slot } => format!("load.slot {} #{slot}", v(table)),
        Op::LoadProto { closure } => format!("load.proto {}", v(closure)),
        Op::LoadArray { table, index, state } => format!("load.array {} [{}] s{}", v(table), v(index), state.0),
        Op::GuardType { value, ty, state } => format!("guard.type {} {ty} s{}", v(value), state.0),
        Op::GuardShape { table, shape, state } => format!("guard.shape {} {shape:x} s{}", v(table), state.0),
        Op::GuardHasMeta { table, state } => format!("guard.meta {} s{}", v(table), state.0),
        Op::GuardNoMeta { table, bit, state } => format!("guard.nometa {} bit{bit} s{}", v(table), state.0),
        Op::GuardEq { value, expected, state } => format!("guard.eq {} {expected:x} s{}", v(value), state.0),
        Op::GuardFalse { cond, state } => format!("guard.false {} s{}", v(cond), state.0),
        Op::GuardNotNil { value, state } => format!("guard.notnil {} s{}", v(value), state.0),
        Op::GuardRange { table, low, high, state } => format!("guard.range {} [{}..{}] s{}", v(table), v(low), v(high), state.0),
        Op::Index { object, key, state } => format!("index {} [{}] s{}", v(object), v(key), state.0),
        Op::NewIndex { object, key, value, state } => format!("newindex {} [{}] = {} s{}", v(object), v(key), v(value), state.0),
        Op::AddSlot { table, key, shape, next, slot, value, state } => format!("add.slot {} [{}] {shape:x}->{next:x} #{slot} = {} s{}", v(table), v(key), v(value), state.0),
        Op::StoreSlot { table, slot, value, state } => format!("store.slot {} #{slot} = {} s{}", v(table), v(value), state.0),
        Op::GuardNode { table, node, nodes, key, state } => format!("guard.node {} #{node}/{nodes} {key:x} s{}", v(table), state.0),
        Op::LoadNode { table, node } => format!("load.node {} #{node}", v(table)),
        Op::StoreNode { table, node, value, state } => format!("store.node {} #{node} = {} s{}", v(table), v(value), state.0),
        Op::Helper(helper) => {
            let kind = match helper.kind {
                HelperKind::Step => "step".to_owned(),
                HelperKind::Fast(index) => format!("fast{index}"),
            };
            format!("helper.{kind} @{} {:?}", helper.pc, helper.args)
        }
        Op::Builtin { function, args, state } => {
            let args: Vec<String> = args.iter().map(v).collect();
            format!("builtin.{} {} s{}", format!("{function:?}").to_lowercase(), args.join(", "), state.0)
        }
        Op::NewTable { hash_log, top, shape, fields, state, after } => {
            let fields: Vec<String> = fields.iter().map(v).collect();
            let shape = shape.map_or(String::new(), |shape| format!(" shape {shape:x} [{}]", fields.join(", ")));
            format!("new.table h{hash_log} top r{top}{shape} s{} after s{}", state.0, after.0)
        }
        Op::Materialize { object } => format!("materialize {}", virtual_text(object)),
        Op::GuardAbsent { table, bit, state } => format!("guard.absent {} bit{bit} s{}", v(table), state.0),
        Op::CallLua { pc, a, b, c } => format!("call @{pc} r{a} b{b} c{c}"),
        Op::Safepoint { pc, weight, state } => format!("safepoint @{pc} x{weight} s{}", state.0),
    }
}
