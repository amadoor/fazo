use super::{
    a64::A64,
    lower::{patch_branch, patch_value},
    masm::{Field, Masm, Stencil},
    templates::{self, ArithKind, CmpKind, Env, HelperAddresses, HelperId, Key, Rhs, TagConstant},
    x64::X64,
};

fn env() -> Env {
    Env {
        table: crate::jit::TableLayout::probe(),
        frames: crate::jit::FrameLayout::probe(),
        depth_limit: 400,
        helpers: HelperAddresses {
            step: 0x1111_2222_3333_4444,
            safe_point: 0x1111_2222_3333_5555,
            misc: 0x1111_2222_3333_6666,
            tier_check: 0x1111_2222_3333_7777,
            fast: std::array::from_fn(|index| 0x7000_0000_0000 + index as u64 * 0x100),
        },
    }
}

fn keys() -> Vec<Key> {
    let mut keys = vec![
        Key::Move,
        Key::LoadConstant,
        Key::LoadInt,
        Key::LoadFloat,
        Key::LoadBool(false),
        Key::LoadBool(true),
        Key::LoadNil,
        Key::Jump,
        Key::BackEdge { charge: false },
        Key::BackEdge { charge: true },
        Key::Charge,
        Key::Unm,
        Key::BNot,
        Key::Not,
        Key::Truthy,
        Key::MoveJump,
        Key::EqTag(TagConstant::Nil),
        Key::EqTag(TagConstant::Bool),
        Key::EqTag(TagConstant::String),
        Key::ForLoop,
        Key::TForLoop,
        Key::RegisterTable,
        Key::UpvalueTable,
        Key::MethodGet,
        Key::MethodStart,
        Key::MethodHop,
        Key::MethodFinish,
        Key::GetUpval,
        Key::SetUpval,
        Key::ReturnCheck,
        Key::ReturnCopy,
        Key::ReturnFill,
        Key::ReturnWide,
        Key::ReturnDone,
        Key::StepStub,
        Key::SharedStep,
        Key::CallCheck,
        Key::CallEnter { upvalues: false },
        Key::CallEnter { upvalues: true },
        Key::CallAnyCheck,
        Key::CallAnyEnter,
        Key::TailCheck,
        Key::ClearAt,
        Key::TailEnter { upvalues: false },
        Key::TailEnter { upvalues: true },
    ];
    for id in [HelperId::Step, HelperId::SafePoint, HelperId::Misc, HelperId::TierCheck] {
        keys.push(Key::Helper(id));
    }
    for which in 0..21 {
        keys.push(Key::Helper(HelperId::Fast(which)));
    }
    let ops = [
        ArithKind::Add,
        ArithKind::Sub,
        ArithKind::Mul,
        ArithKind::BAnd,
        ArithKind::BOr,
        ArithKind::BXor,
        ArithKind::Div,
        ArithKind::Mod,
        ArithKind::IDiv,
    ];
    for op in ops {
        for rhs in [Rhs::Reg, Rhs::KInt, Rhs::KFloat] {
            keys.push(Key::Arith { op, rhs });
        }
    }
    keys.push(Key::Arith { op: ArithKind::Add, rhs: Rhs::Imm });
    for cmp in [CmpKind::Eq, CmpKind::Lt, CmpKind::Le] {
        keys.push(Key::Compare { cmp, rhs: Rhs::Reg, int: true, float: true, default_false: false });
    }
    keys.push(Key::Compare { cmp: CmpKind::Eq, rhs: Rhs::KInt, int: true, float: false, default_false: false });
    keys.push(Key::Compare { cmp: CmpKind::Eq, rhs: Rhs::Imm, int: true, float: true, default_false: true });
    for cmp in [CmpKind::Lt, CmpKind::Le, CmpKind::Gt, CmpKind::Ge] {
        keys.push(Key::Compare { cmp, rhs: Rhs::Imm, int: true, float: false, default_false: false });
        keys.push(Key::Compare { cmp, rhs: Rhs::Imm, int: false, float: true, default_false: false });
    }
    for constant_key in [false, true] {
        keys.push(Key::ArrayGet { constant_key });
        for constant_value in [false, true] {
            keys.push(Key::ArraySet { constant_key, constant_value });
        }
    }
    for absent in [false, true] {
        for method in [false, true] {
            keys.push(Key::FieldGet { absent, method });
        }
    }
    for constant in [false, true] {
        keys.push(Key::FieldSet { constant });
        keys.push(Key::FieldAdd { constant });
        keys.push(Key::HashSet { constant_value: constant });
    }
    keys.push(Key::HashGet);
    keys
}

fn stencils(masm: fn() -> Box<dyn Masm>) -> Vec<(String, Stencil)> {
    let env = env();
    let mut stencils = Vec::new();
    for key in keys() {
        let mut m = masm();
        templates::build(key, m.as_mut(), &env);
        stencils.push((format!("{key:?}"), m.finish()));
    }
    stencils
}

fn x64_masm() -> Box<dyn Masm> {
    Box::new(X64::new())
}

fn a64_masm() -> Box<dyn Masm> {
    Box::new(A64::new())
}

fn x64_frames() -> Vec<(String, Stencil)> {
    use super::{masm::ext, x64::frame};
    let reserved = frame::reserved(0);
    let prologue = frame::prologue(reserved, 8, 16);
    vec![
        ("prologue".into(), Stencil { bytes: prologue.bytes, holes: Vec::new() }),
        ("epilogue".into(), Stencil { bytes: frame::epilogue(reserved), holes: Vec::new() }),
        ("negate".into(), Stencil { bytes: frame::negate_exit(), holes: Vec::new() }),
        ("dispatch".into(), frame::dispatch(ext::EXIT, ext::TARGET)),
    ]
}

fn a64_frames() -> Vec<(String, Stencil)> {
    use super::{a64::frame, masm::ext};
    let reserved = frame::reserved(0);
    let prologue = frame::prologue(reserved, 8, 16);
    vec![
        ("prologue".into(), Stencil { bytes: prologue.bytes, holes: Vec::new() }),
        ("epilogue".into(), Stencil { bytes: frame::epilogue(reserved), holes: Vec::new() }),
        ("negate".into(), Stencil { bytes: frame::negate_exit(), holes: Vec::new() }),
        ("dispatch".into(), frame::dispatch(ext::EXIT, ext::TARGET)),
    ]
}

mod x86 {
    use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter, Mnemonic, OpKind};

    use super::*;

    fn decode(bytes: &[u8]) -> Vec<(usize, Instruction)> {
        let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
        let mut out = Vec::new();
        while decoder.can_decode() {
            let at = decoder.position();
            let instruction = decoder.decode();
            out.push((at, instruction));
        }
        out
    }

    fn text(instruction: &Instruction) -> String {
        let mut output = String::new();
        IntelFormatter::new().format(instruction, &mut output);
        output
    }

    fn containing(bytes: &[u8], at: usize) -> Instruction {
        decode(bytes)
            .into_iter()
            .find(|(start, instruction)| *start <= at && at + 4 <= start + instruction.len())
            .map(|(_, instruction)| instruction)
            .expect("a hole lies inside one instruction")
    }

    fn immediates(instruction: &Instruction) -> Vec<i64> {
        let mut values = vec![instruction.memory_displacement64() as i64, instruction.memory_displacement32() as i32 as i64];
        for operand in 0..instruction.op_count() {
            match instruction.op_kind(operand) {
                OpKind::Immediate32 | OpKind::Immediate32to64 | OpKind::Immediate64 | OpKind::Immediate8to64 => {
                    values.push(instruction.immediate(operand) as i64);
                    values.push(instruction.immediate(operand) as u32 as i32 as i64);
                }
                _ => {}
            }
        }
        values
    }

    fn check(name: &str, stencil: &Stencil, mnemonics: &mut std::collections::BTreeSet<String>) {
        let decoded = decode(&stencil.bytes);
        let length: usize = decoded.iter().map(|(_, instruction)| instruction.len()).sum();
        assert_eq!(length, stencil.bytes.len(), "{name}: every byte decodes");
        for (at, instruction) in &decoded {
            assert!(!instruction.is_invalid(), "{name}: an invalid instruction at {at}");
            mnemonics.insert(format!("{:?}", instruction.mnemonic()).to_lowercase());
        }
        for hole in &stencil.holes {
            let at = hole.offset as usize;
            for value in [0x1230_i64, 0x7ff0, -0x40, 0x10] {
                let mut bytes = stencil.bytes.clone();
                match hole.field {
                    Field::Le32 => patch_value(&mut bytes, at, hole.field, value),
                    Field::Le64 => patch_value(&mut bytes, at, hole.field, value << 20),
                    Field::Rel32 => patch_branch(&mut bytes, at, hole.field, value + 0x1000).expect("in range"),
                    other => panic!("{name}: x86-64 has no {other:?} field"),
                }
                let instruction = containing(&bytes, at);
                let shown = text(&instruction);
                match hole.field {
                    Field::Le32 => assert!(immediates(&instruction).contains(&value), "{name}: {value:#x} reads back from `{shown}`"),
                    Field::Le64 => assert_eq!(instruction.immediate64() as i64, value << 20, "{name}: `{shown}`"),
                    _ => {
                        assert!(instruction.op_kind(0) == OpKind::NearBranch64 || instruction.mnemonic() == Mnemonic::Lea, "{name}: `{shown}` branches");
                        let target = if instruction.mnemonic() == Mnemonic::Lea { instruction.ip_rel_memory_address() } else { instruction.near_branch64() };
                        assert_eq!(target as i64, value + 0x1000, "{name}: `{shown}` reaches its target");
                    }
                }
            }
        }
    }

    #[test]
    fn every_x64_template_decodes_and_its_holes_read_back() {
        let mut mnemonics = std::collections::BTreeSet::new();
        let all: Vec<_> = stencils(x64_masm).into_iter().chain(x64_frames()).collect();
        let holes: usize = all.iter().map(|(_, stencil)| stencil.holes.len()).sum();
        for (name, stencil) in &all {
            check(name, stencil, &mut mnemonics);
        }
        eprintln!("x86-64: {} stencils, {holes} holes, mnemonics {mnemonics:?}", all.len());
    }
}

mod arm {
    use yaxpeax_arch::{Decoder, U8Reader};
    use yaxpeax_arm::armv8::a64::{InstDecoder, Instruction};

    use super::*;

    fn decode_word(word: u32) -> Option<Instruction> {
        let bytes = word.to_le_bytes();
        let mut reader = U8Reader::new(&bytes);
        InstDecoder::default().decode(&mut reader).ok()
    }

    fn word(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
    }

    fn check(name: &str, stencil: &Stencil, mnemonics: &mut std::collections::BTreeSet<String>) {
        assert_eq!(stencil.bytes.len() % 4, 0, "{name}: whole instructions");
        for at in (0..stencil.bytes.len()).step_by(4) {
            let instruction = decode_word(word(&stencil.bytes, at)).unwrap_or_else(|| panic!("{name}: {:#010x} at {at} decodes", word(&stencil.bytes, at)));
            let shown = instruction.to_string();
            mnemonics.insert(shown.split_whitespace().next().unwrap_or_default().to_string());
        }
        for hole in &stencil.holes {
            let at = hole.offset as usize;
            for step in [1_i64, 3, 0x45] {
                let mut bytes = stencil.bytes.clone();
                let expected = match hole.field {
                    Field::Imm12 { scale } => {
                        let value = step << scale;
                        patch_value(&mut bytes, at, hole.field, value);
                        format!("#{value:#x}")
                    }
                    Field::Imm16 { shift } => {
                        let piece = 0xbe00 + step;
                        patch_value(&mut bytes, at, hole.field, piece << shift);
                        format!("#{piece:#x}")
                    }
                    Field::Branch26 | Field::Branch19 | Field::Branch14 | Field::Adr => {
                        let target = at as i64 + step * 4;
                        patch_branch(&mut bytes, at, hole.field, target).expect("in range");
                        format!("$+{:#x}", step * 4)
                    }
                    other => panic!("{name}: AArch64 has no {other:?} field"),
                };
                let shown = decode_word(word(&bytes, at)).map(|instruction| instruction.to_string()).unwrap_or_default();
                assert!(shown.contains(&expected), "{name}: {:?} at {at} reads back {expected} from `{shown}`", hole.field);
            }
        }
    }

    #[test]
    fn every_a64_template_decodes_and_its_holes_read_back() {
        let mut mnemonics = std::collections::BTreeSet::new();
        let all: Vec<_> = stencils(a64_masm).into_iter().chain(a64_frames()).collect();
        let holes: usize = all.iter().map(|(_, stencil)| stencil.holes.len()).sum();
        for (name, stencil) in &all {
            check(name, stencil, &mut mnemonics);
        }
        eprintln!("AArch64: {} stencils, {holes} holes, mnemonics {mnemonics:?}", all.len());
    }
}

#[test]
fn both_frames_have_unwind_trailers() {
    use super::unwind::{Prologue, trailer};
    let reserved = super::a64::frame::reserved(0);
    let prologue = super::a64::frame::prologue(reserved, 8, 16);
    let a64 = Prologue::A64 {
        save_frame: prologue.save_frame,
        set_frame: prologue.set_frame,
        pairs: prologue.pairs,
        reserved,
    };
    let reserved = super::x64::frame::reserved(0);
    let prologue = super::x64::frame::prologue(reserved, 8, 16);
    let x64 = Prologue::X64 {
        push_rbp: u32::from(prologue.push_rbp),
        set_rbp: u32::from(prologue.set_rbp),
        pushes: std::array::from_fn(|index| (u32::from(prologue.pushes[index]), super::x64::frame::SAVED[index] as u8)),
        alloc: u32::from(prologue.alloc),
        reserved,
    };
    for prologue in [a64, x64] {
        let trailer = trailer(&prologue, 4096, 4096).expect("the frame has unwind information");
        assert!(!trailer.bytes.is_empty() && trailer.bytes.len().is_multiple_of(4), "{prologue:?}");
    }
}

#[test]
fn every_key_has_its_own_index() {
    let mut seen = vec![false; templates::KEY_COUNT];
    for key in keys() {
        let index = key.index();
        assert!(index < templates::KEY_COUNT && !seen[index], "{key:?} has index {index}");
        seen[index] = true;
    }
}
