//! Follows CPython's specializing interpreter (PEP 659, `Python/specialize.c`).
//! Only the executed copy changes; PUC's code, dumps and error names stay as they are.

use std::cell::Cell;

use super::ic::{ABSENT, Entry, InlineCache};
use crate::{Vm, arena::Ref, instruction::Instruction, opcode::Opcode, proto::Proto};

macro_rules! dispatch {
    (generic: [$($generic:ident,)*] specialized: [$($form:ident => $of:ident,)*]) => {
        /// Specialized forms take the opcode values Lua leaves unused.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub(crate) enum Dispatch {
            $($generic,)*
            $($form,)*
        }

        const _: () = {
            $(assert!(Dispatch::$generic as u8 == Opcode::$generic as u8);)*
        };

        impl Dispatch {
            pub(crate) const ALL: &'static [Dispatch] = &[$(Dispatch::$generic,)* $(Dispatch::$form,)*];
            pub(crate) const COUNT: usize = Dispatch::ALL.len();

            #[inline(always)]
            pub(crate) fn generic(self) -> Opcode {
                match self {
                    $(Dispatch::$generic => Opcode::$generic,)*
                    $(Dispatch::$form => Opcode::$of,)*
                }
            }
        }
    };
}

dispatch! {
    generic: [
    Move, LoadI, LoadF, LoadK, LoadKx, LoadFalse, LFalseSkip, LoadTrue, LoadNil, GetUpval, SetUpval,
    GetTabUp, GetTable, GetI, GetField, SetTabUp, SetTable, SetI, SetField, NewTable, SelfOp, AddI,
    AddK, SubK, MulK, ModK, PowK, DivK, IDivK, BAndK, BOrK, BXorK, ShrI, ShlI, Add, Sub, Mul, Mod,
    Pow, Div, IDiv, BAnd, BOr, BXor, Shl, Shr, MmBin, MmBinI, MmBinK, Unm, BNot, Not, Len, Concat,
    Close, Tbc, Jmp, Eq, Lt, Le, EqK, EqI, LtI, LeI, GtI, GeI, Test, TestSet, Call, TailCall,
    Return, Return0, Return1, ForLoop, ForPrep, TForPrep, TForCall, TForLoop, SetList, Closure,
    Vararg, GetVarg, ErrNil, VarargPrep, ExtraArg,
    ]
    specialized: [
    GetFieldShape => GetField,
    GetFieldShapes => GetField,
    GetFieldMethod => GetField,
    GetFieldAbsent => GetField,
    GetTabUpShape => GetTabUp,
    SelfShape => SelfOp,
    SelfMethod => SelfOp,
    SetFieldShape => SetField,
    SetFieldAdd => SetField,
    SetTabUpShape => SetTabUp,
    GetTableInt => GetTable,
    GetTableAny => GetTable,
    GetIInt => GetI,
    GetIAny => GetI,
    SetTableInt => SetTable,
    SetTableAny => SetTable,
    SetIInt => SetI,
    SetIAny => SetI,
    NewTableOpen => NewTable,
    ]
}

const _: () = assert!(Dispatch::COUNT <= 1 << 7);

const OPCODE_MASK: u32 = 0x7f;

pub(crate) const MISS_BUDGET: u8 = 16;

/// Rebuilt from `code` and never saved.
#[derive(Clone, Debug, Default)]
pub(crate) struct Words {
    sites: Box<[Site]>,
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub(crate) struct Site {
    pub(crate) word: Cell<u32>,
    pub(crate) misses: Cell<u8>,
}

impl Words {
    pub(crate) fn of(code: &[Instruction]) -> Self {
        let words = Self {
            sites: code
                .iter()
                .map(|&instruction| Site {
                    word: Cell::new(crate::encoding::encode(instruction)),
                    misses: Cell::new(0),
                })
                .collect(),
        };
        for pc in open_constructors(code) {
            words.rewrite(pc, Dispatch::NewTableOpen);
        }
        words
    }

    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn opens_list(&self, pc: usize) -> bool {
        self.sites.get(pc).is_some_and(|site| super::window::opcode_of(site.word.get()) == Dispatch::NewTableOpen)
    }

    #[cfg(debug_assertions)]
    pub(crate) fn len(&self) -> usize {
        self.sites.len()
    }

    pub(crate) fn as_ptr(&self) -> *const Site {
        self.sites.as_ptr()
    }

    /// A specialized form starts with a full miss budget.
    pub(crate) fn rewrite(&self, pc: usize, form: Dispatch) {
        let Some(site) = self.sites.get(pc) else {
            return;
        };
        site.word.set((site.word.get() & !OPCODE_MASK) | u32::from(form as u8));
        site.misses.set(MISS_BUDGET);
    }

    #[cfg(test)]
    pub(crate) fn form(&self, pc: usize) -> Dispatch {
        super::window::opcode_of(self.sites[pc].word.get())
    }
}

/// The size hint of these `NEWTABLE`s grows once more at run time, since their `SETLIST` count is open.
fn open_constructors(code: &[Instruction]) -> impl Iterator<Item = usize> + '_ {
    code.iter().enumerate().filter_map(|(pc, instruction)| match *instruction {
        Instruction::SetList { a, b: 0, .. } => code[..pc]
            .iter()
            .rposition(|earlier| matches!(*earlier, Instruction::NewTable { a: target, .. } if target == a)),
        _ => None,
    })
}

fn form_for(instruction: Instruction, cache: &InlineCache) -> Dispatch {
    match (instruction, cache) {
        (Instruction::GetField { .. }, InlineCache::Monomorphic(entry)) if present(entry) => Dispatch::GetFieldShape,
        (Instruction::GetField { .. }, InlineCache::Polymorphic(entries)) if entries.iter().all(present) => Dispatch::GetFieldShapes,
        (Instruction::GetField { .. }, InlineCache::Method(_)) => Dispatch::GetFieldMethod,
        (Instruction::GetField { .. }, InlineCache::Monomorphic(entry)) if !entry.adds_key() => Dispatch::GetFieldAbsent,
        (Instruction::GetTabUp { .. }, InlineCache::Monomorphic(entry)) if present(entry) => Dispatch::GetTabUpShape,
        (Instruction::SelfOp { .. }, InlineCache::Monomorphic(entry)) if present(entry) => Dispatch::SelfShape,
        (Instruction::SelfOp { .. }, InlineCache::Method(_)) => Dispatch::SelfMethod,
        (Instruction::SetField { .. }, InlineCache::Monomorphic(entry)) if present(entry) => Dispatch::SetFieldShape,
        (Instruction::SetField { .. }, InlineCache::Monomorphic(entry)) if entry.adds_key() => Dispatch::SetFieldAdd,
        (Instruction::SetTabUp { .. }, InlineCache::Monomorphic(entry)) if present(entry) => Dispatch::SetTabUpShape,
        _ => generic_form(instruction.opcode()),
    }
}

fn present(entry: &Entry) -> bool {
    !entry.adds_key() && entry.slot != ABSENT
}

fn generic_form(opcode: Opcode) -> Dispatch {
    Dispatch::ALL[usize::from(opcode as u8)]
}

impl Vm {
    pub(crate) fn quicken_site(&self, proto: Ref<Proto>, site: usize) {
        let function = self.heap.protos.live(proto);
        let (Some(&instruction), Some(cache)) = (function.code.get(site), function.exec.feedback.get(site)) else {
            return;
        };
        function.exec.words.rewrite(site, form_for(instruction, cache));
    }
}

#[cfg(test)]
mod tests {
    use super::Dispatch;
    use crate::{Host, Options, Value, Vm, opcode::Opcode};

    const PROGRAM: &[u8] = b"function get(t) return t.x end
function index(t, k) return t[k] end
point = {x = 1, y = 2}
total = 0
for _ = 1, 50 do total = total + get(point) end";

    fn interpreter() -> Options {
        let mut options = Options::default();
        assert!(options.set_extension(b"tier=interpreter"));
        options
    }

    fn form_of(vm: &mut Vm, name: &str, opcode: Opcode) -> Dispatch {
        let Value::Closure(closure) = vm.global(name) else {
            panic!("{name} is a Lua function");
        };
        let function = vm.proto_of(closure);
        let site = function.code.iter().position(|instruction| instruction.opcode() == opcode).expect("the function has the instruction");
        function.exec.words.form(site)
    }

    fn total(vm: &mut Vm) -> i64 {
        let Value::Integer(total) = vm.global("total") else {
            panic!("total is an integer");
        };
        total
    }

    #[test]
    fn a_field_read_quickens_to_its_shape_and_reverts_when_the_site_turns_megamorphic() {
        let mut vm = Vm::new(interpreter());
        vm.run_source(b"=quicken", PROGRAM).expect("the program runs");
        assert_eq!(form_of(&mut vm, "get", Opcode::GetField), Dispatch::GetFieldShape);
        assert_eq!(total(&mut vm), 50);

        vm.run_source(b"=shapes", b"total = 0 for i = 1, 8 do local t = {} t['k' .. i] = 0 t.x = i total = total + get(t) end")
            .expect("the shapes run");
        assert_eq!(form_of(&mut vm, "get", Opcode::GetField).generic(), Opcode::GetField);
        assert_ne!(form_of(&mut vm, "get", Opcode::GetField), Dispatch::GetFieldShape);
        assert_eq!(total(&mut vm), 36);
    }

    #[test]
    fn an_integer_index_falls_back_to_the_generic_read_after_its_misses() {
        let mut vm = Vm::new(interpreter());
        vm.run_source(b"=quicken", PROGRAM).expect("the program runs");
        vm.run_source(b"=ints", b"local t = {10, 20} total = index(t, 1) + index(t, 2)").expect("the reads run");
        assert_eq!(form_of(&mut vm, "index", Opcode::GetTable), Dispatch::GetTableInt);
        assert_eq!(total(&mut vm), 30);

        vm.run_source(b"=mixed", b"local t = {[0] = 1, 2, s = 3} total = 0 for _ = 1, 40 do total = total + index(t, 0) + index(t, 1) + index(t, 's') end")
            .expect("the mixed reads run");
        assert_eq!(form_of(&mut vm, "index", Opcode::GetTable), Dispatch::GetTableAny);
        assert_eq!(total(&mut vm), 240);
    }

    #[test]
    fn a_restored_vm_rebuilds_its_words_and_gives_the_same_results() {
        let mut vm = Vm::new(interpreter());
        vm.run_source(b"=quicken", PROGRAM).expect("the program runs");
        let bytes = vm.snapshot().expect("the VM is at an instruction boundary");
        let mut restored = Vm::restore(&bytes, interpreter(), Host::captured()).expect("the snapshot restores");
        assert_eq!(form_of(&mut restored, "get", Opcode::GetField), Dispatch::GetField);

        let more = b"for _ = 1, 50 do total = total + get(point) end";
        vm.run_source(b"=more", more).expect("the original runs on");
        restored.run_source(b"=more", more).expect("the restored VM runs on");
        assert_eq!(total(&mut vm), total(&mut restored));
        assert_eq!(form_of(&mut restored, "get", Opcode::GetField), Dispatch::GetFieldShape);
    }
}
