use crate::{heap::Heap, instruction::Instruction, proto::Proto, value::Value};

fn main_proto(source: &str) -> (Proto, Heap) {
    let mut heap = Heap::default();
    let proto = super::compile(&mut heap, b"=test", source.as_bytes(), None).expect("the source compiles");
    let proto = heap.protos.get(proto).clone();
    (proto, heap)
}

#[test]
fn a_local_initialized_with_a_constant_is_loaded_then_returned() {
    let (proto, _) = main_proto("local a = 1\nreturn a");
    assert_eq!(
        proto.code,
        vec![
            Instruction::VarargPrep { a: 0 },
            Instruction::LoadI { a: 0, sbx: 1 },
            Instruction::Return { a: 0, b: 2, c: 1, k: false },
            Instruction::Return { a: 1, b: 1, c: 1, k: false },
        ]
    );
}

#[test]
fn adding_a_small_integer_uses_the_immediate_form() {
    let (proto, _) = main_proto("local x = 0\nreturn x + 1");
    assert_eq!(
        proto.code,
        vec![
            Instruction::VarargPrep { a: 0 },
            Instruction::LoadI { a: 0, sbx: 0 },
            Instruction::AddI { a: 1, b: 0, sc: 1 },
            Instruction::MmBinI {
                a: 0,
                sb: 1,
                c: 6,
                k: false,
            },
            Instruction::Return { a: 1, b: 2, c: 1, k: false },
            Instruction::Return { a: 1, b: 1, c: 1, k: false },
        ]
    );
}

#[test]
fn a_global_is_read_through_the_environment_upvalue() {
    let (proto, heap) = main_proto("return print");
    assert_eq!(
        proto.code,
        vec![
            Instruction::VarargPrep { a: 0 },
            Instruction::GetTabUp { a: 0, b: 0, c: 0 },
            Instruction::Return { a: 0, b: 2, c: 1, k: false },
            Instruction::Return { a: 0, b: 1, c: 1, k: false },
        ]
    );
    let Value::String(name) = proto.constants[0] else {
        panic!("the key of a global is a string constant");
    };
    assert_eq!(heap.bytes(name), b"print");
}

#[test]
fn an_if_test_against_an_integer_jumps_over_the_body_when_false() {
    let (proto, _) = main_proto("local a = 1\nif a < 10 then return 1 end\nreturn 2");
    assert_eq!(
        proto.code[2],
        Instruction::LtI {
            a: 0,
            sb: 10,
            c: 0,
            k: false,
        }
    );
    assert_eq!(proto.code[3], Instruction::Jmp { sj: 2 });
    assert_eq!(proto.code[4], Instruction::LoadI { a: 1, sbx: 1 });
}

#[test]
fn an_empty_numeric_for_prepares_and_loops_over_its_own_instruction() {
    let (proto, _) = main_proto("for i = 1, 3 do end");
    assert_eq!(proto.code[1], Instruction::LoadI { a: 0, sbx: 1 });
    assert_eq!(proto.code[2], Instruction::LoadI { a: 1, sbx: 3 });
    assert_eq!(proto.code[3], Instruction::LoadI { a: 2, sbx: 1 });
    assert_eq!(proto.code[4], Instruction::ForPrep { a: 0, bx: 0 });
    assert_eq!(proto.code[5], Instruction::ForLoop { a: 0, bx: 1 });
}

#[test]
fn constant_arithmetic_is_folded() {
    let (proto, _) = main_proto("return 2 * 3");
    assert_eq!(proto.code[1], Instruction::LoadI { a: 0, sbx: 6 });
}

#[test]
fn a_folded_negative_zero_stays_a_runtime_negation_so_its_sign_survives() {
    let (proto, _) = main_proto("return -0.0");
    assert_eq!(proto.code[1], Instruction::LoadF { a: 0, sbx: 0 });
    assert_eq!(
        proto.code[2],
        Instruction::Unm { a: 0, b: 0 }
    );
}
