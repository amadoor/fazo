use super::{
    alloc::{Alloc, Have, Moves, Want, part, resolve},
    asm::{Alu, Asm, Cond, Division, FAlu, FCond, Width},
    live::Liveness,
    regs::{Class, PReg, RegSet, Registers},
};
use crate::jit::ssa::ir::{BinOp, Cc, Const, Edge, Func, Op, Rep, Term, Type, Val};

const GP: [u8; 4] = [1, 2, 3, 4];
const FP: [u8; 2] = [0, 1];
const SMALL: Registers = Registers { gp: &GP, fp: &FP, callee_saved: RegSet(1 << 4) };

struct Machine {
    regs: [u64; 64],
    slots: [u64; 16],
    scratch: u64,
    log: Vec<String>,
}

impl Machine {
    fn new() -> Self {
        Self { regs: [0; 64], slots: [0; 16], scratch: 0, log: Vec::new() }
    }
}

impl Moves for Machine {
    fn copy(&mut self, dst: PReg, src: PReg) {
        self.regs[dst.index()] = self.regs[src.index()];
        self.log.push(format!("copy r{} r{}", dst.index(), src.index()));
    }

    fn spill(&mut self, slot: u32, src: PReg) {
        self.slots[slot as usize] = self.regs[src.index()];
        self.log.push(format!("spill s{slot} r{}", src.index()));
    }

    fn reload(&mut self, dst: PReg, slot: u32) {
        self.regs[dst.index()] = self.slots[slot as usize];
        self.log.push(format!("reload r{} s{slot}", dst.index()));
    }

    fn constant(&mut self, dst: PReg, bits: u64) {
        self.regs[dst.index()] = bits;
        self.log.push(format!("constant r{} {bits}", dst.index()));
    }

    fn constant_slot(&mut self, slot: u32, bits: u64) {
        self.slots[slot as usize] = bits;
        self.log.push(format!("constant s{slot} {bits}"));
    }

    fn slot_copy(&mut self, _class: Class, dst: u32, src: u32) {
        self.slots[dst as usize] = self.slots[src as usize];
        self.log.push(format!("slot s{dst} s{src}"));
    }

    fn save_scratch(&mut self, src: PReg) {
        self.scratch = self.regs[src.index()];
    }

    fn restore_scratch(&mut self, dst: PReg) {
        self.regs[dst.index()] = self.scratch;
    }
}

fn gp(hw: u8) -> PReg {
    PReg::gp(hw)
}

fn want(reg: u8) -> Want {
    Want { class: Class::Gp, reg: Some(gp(reg)), slot: None }
}

fn have(reg: u8) -> Have {
    Have { reg: Some(gp(reg)), ..Have::default() }
}

fn moved(pairs: &[(Want, Have)]) -> Machine {
    let mut machine = Machine::new();
    for (index, value) in machine.regs.iter_mut().enumerate() {
        *value = 100 + index as u64;
    }
    for (index, value) in machine.slots.iter_mut().enumerate() {
        *value = 200 + index as u64;
    }
    resolve(pairs, &mut machine);
    machine
}

#[test]
fn edge_moves_resolve_swaps_rotations_and_fan_out() {
    let cases: [&[(u8, u8)]; 5] = [
        &[(1, 2), (2, 1)],
        &[(1, 2), (2, 3), (3, 1)],
        &[(1, 2), (2, 1), (3, 4), (4, 3)],
        &[(1, 2), (3, 2), (2, 4)],
        &[(1, 2), (2, 3), (3, 1), (4, 1)],
    ];
    for case in cases {
        let pairs: Vec<(Want, Have)> = case.iter().map(|&(dst, src)| (want(dst), have(src))).collect();
        let machine = moved(&pairs);
        for &(dst, src) in case {
            assert_eq!(machine.regs[usize::from(dst)], 100 + u64::from(src), "{case:?}: r{dst} gets r{src}");
        }
    }
}

#[test]
fn edge_moves_store_slots_before_registers_change_and_load_last() {
    let pairs = [
        (Want { class: Class::Gp, reg: None, slot: Some(3) }, have(1)),
        (want(1), Have { slot: Some(5), ..Have::default() }),
        (want(2), Have { constant: Some(7), ..Have::default() }),
        (Want { class: Class::Gp, reg: None, slot: Some(4) }, Have { slot: Some(6), ..Have::default() }),
    ];
    let machine = moved(&pairs);
    assert_eq!(machine.slots[3], 101);
    assert_eq!(machine.regs[1], 205);
    assert_eq!(machine.regs[2], 7);
    assert_eq!(machine.slots[4], 206);
    assert_eq!(machine.log.first().map(String::as_str), Some("spill s3 r1"));
}

fn gp_alloc(values: usize) -> Alloc {
    Alloc::new(SMALL, vec![Class::Gp; values * 2])
}

#[test]
fn a_full_class_spills_the_value_used_last_and_reloads_it_on_use() {
    let mut alloc = gp_alloc(6);
    let mut machine = Machine::new();
    for value in 0..4 {
        alloc.define(part(Val(value), 0), &mut machine);
        alloc.end_node(&[]);
    }
    alloc.last = vec![10, 50, 20, 30, 40, 40];
    let fifth = alloc.define(part(Val(4), 0), &mut machine);
    assert_eq!(alloc.reg_of(part(Val(1), 0)), None, "the value used last leaves its register");
    assert_eq!(Some(fifth), Some(gp(2)));
    assert_eq!(machine.log, ["spill s0 r2"]);
    alloc.end_node(&[]);
    let back = alloc.use_reg(part(Val(1), 0), &mut machine);
    assert_eq!(machine.log.last().map(String::as_str), Some(&*format!("reload r{} s0", back.index())));
}

#[test]
fn a_spilled_value_is_stored_once() {
    let mut alloc = gp_alloc(6);
    let mut machine = Machine::new();
    for value in 0..4 {
        alloc.define(part(Val(value), 0), &mut machine);
        alloc.end_node(&[]);
    }
    alloc.last = vec![10, 50, 20, 30, 40, 40];
    alloc.define(part(Val(4), 0), &mut machine);
    alloc.end_node(&[Val(4)]);
    alloc.use_reg(part(Val(1), 0), &mut machine);
    alloc.end_node(&[]);
    alloc.last[1] = 60;
    alloc.define(part(Val(5), 0), &mut machine);
    let spills = machine.log.iter().filter(|line| line.starts_with("spill")).count();
    assert_eq!(spills, 1, "{:?}", machine.log);
}

#[test]
fn operands_and_temporaries_of_one_node_never_share_a_register() {
    let mut alloc = gp_alloc(6);
    let mut machine = Machine::new();
    for value in 0..4 {
        alloc.define(part(Val(value), 0), &mut machine);
        alloc.end_node(&[]);
    }
    let left = alloc.use_reg(part(Val(0), 0), &mut machine);
    let right = alloc.use_reg(part(Val(1), 0), &mut machine);
    let temp = alloc.temp(Class::Gp, &mut machine);
    let out = alloc.define(part(Val(4), 0), &mut machine);
    let all = [left, right, temp, out];
    for (index, reg) in all.iter().enumerate() {
        assert!(!all[index + 1..].contains(reg), "{all:?}");
    }
}

#[test]
fn a_call_saves_only_caller_saved_registers_whose_values_live_on() {
    let mut alloc = gp_alloc(6);
    let mut machine = Machine::new();
    for value in 0..4 {
        alloc.define(part(Val(value), 0), &mut machine);
        alloc.end_node(&[]);
    }
    let saved = alloc.across_call(|part| part != super::alloc::part(Val(2), 0));
    let regs: Vec<u8> = saved.iter().map(|&(reg, _, _)| reg.hw).collect();
    assert!(!regs.contains(&4), "r4 is callee-saved");
    assert!(!regs.contains(&3), "Val(2) dies at the call");
    assert_eq!(regs, [1, 2]);
}

/// A counted loop that keeps `table` live across it.
fn counted_loop() -> (Func, Val, Val) {
    let mut func = Func::default();
    let entry = func.new_block();
    let header = func.new_block();
    let latch = func.new_block();
    let exit = func.new_block();
    func.entry = Some(entry);
    let table = func.append(entry, Op::LoadReg { slot: 0 }, Type::ANY, Rep::Boxed);
    let start = func.append(entry, Op::Const(Const::Int(1)), Type::INT, Rep::Int);
    let count = func.append(entry, Op::LoadReg { slot: 1 }, Type::ANY, Rep::Boxed);
    let unboxed = func.append(entry, Op::Unary(crate::jit::ssa::ir::UnOp::Unbox, count), Type::INT, Rep::Int);
    func.block_mut(entry).term = Term::Jump(Edge { block: header, args: vec![start, unboxed] });
    let index = func.add_param(header, Type::INT, Rep::Int);
    let remaining = func.add_param(header, Type::INT, Rep::Int);
    let zero = func.append(header, Op::Const(Const::Int(0)), Type::INT, Rep::Int);
    let done = func.append(header, Op::Binary(BinOp::ICmp(Cc::Eq), remaining, zero), Type::BOOL, Rep::Bool);
    func.block_mut(header).term = Term::Branch { cond: done, then: Edge::to(exit), other: Edge::to(latch) };
    let one = func.append(latch, Op::Const(Const::Int(1)), Type::INT, Rep::Int);
    let next_count = func.append(latch, Op::Binary(BinOp::ISub, remaining, one), Type::INT, Rep::Int);
    let next_index = func.append(latch, Op::Binary(BinOp::IAdd, index, one), Type::INT, Rep::Int);
    func.block_mut(latch).term = Term::Jump(Edge { block: header, args: vec![next_index, next_count] });
    func.append(exit, Op::StoreReg { slot: 2, value: table }, Type::NONE, Rep::None);
    func.block_mut(exit).term = Term::Return { pc: 9, first: 0, count: 0, close: false };
    (func, table, done)
}

#[test]
fn liveness_carries_a_value_through_the_loop_and_fuses_the_exit_test() {
    let (func, table, done) = counted_loop();
    let live = Liveness::new(&func);
    let blocks: Vec<_> = live.order.clone();
    for &block in &blocks[1..] {
        assert!(live.is_live_in(block, table), "the table lives through {block:?}");
    }
    assert!(live.fused[done.0 as usize], "the comparison is tested at its branch");
    let header = blocks[1];
    let params = func.block(header).params.clone();
    assert!(params.iter().all(|&param| !live.is_live_in(header, param)), "parameters are defined by the header");
    let constants = func.insts.iter().enumerate().filter(|(_, inst)| matches!(inst.op, Op::Const(_)));
    for (index, _) in constants {
        assert!(blocks.iter().all(|&block| !live.is_live_in(block, Val(index as u32))), "constants are rebuilt at use");
    }
}

#[test]
fn deaths_mark_each_value_at_its_last_use_in_the_block() {
    let (func, _, _) = counted_loop();
    let live = Liveness::new(&func);
    let latch = live.order.iter().copied().find(|&block| matches!(func.block(block).term, Term::Jump(_)) && func.block(block).params.is_empty() && block != func.entry.expect("an entry")).expect("the latch");
    let mut last = vec![u32::MAX; func.insts.len()];
    let dying = live.deaths(&func, latch, &mut last);
    let header = live.order[1];
    let remaining = func.block(header).params[1];
    let index = func.block(header).params[0];
    assert!(dying[1].contains(&remaining), "n dies at n - 1, after the constant at 0");
    assert!(dying[2].contains(&index), "i dies at i + 1");
}

mod x64_encoding {
    use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter, Mnemonic, OpKind, Register};

    use super::*;
    use crate::jit::ssa::emit::x64::X64;

    fn listing(build: impl FnOnce(&mut X64)) -> Vec<String> {
        let mut asm = X64::new();
        build(&mut asm);
        let code = asm.finish().expect("the code assembles");
        let mut decoder = Decoder::with_ip(64, &code, 0, DecoderOptions::NONE);
        let mut formatter = IntelFormatter::new();
        let mut lines = Vec::new();
        for instruction in &mut decoder {
            let mut text = String::new();
            formatter.format(&instruction, &mut text);
            lines.push(text);
        }
        lines
    }

    #[test]
    fn three_operand_integer_forms_keep_both_sources() {
        assert_eq!(listing(|a| a.alu(Alu::Sub, 2, 1, 2)), ["mov r11,rdx", "mov rdx,rcx", "sub rdx,r11"]);
        assert_eq!(listing(|a| a.alu(Alu::Add, 2, 1, 2)), ["add rdx,rcx"]);
        assert_eq!(listing(|a| a.alu_imm(Alu::Mul, 6, 7, 24)), ["imul rsi,rdi,18h"]);
    }

    #[test]
    fn floor_division_keeps_rdx_and_the_divisor() {
        let lines = listing(|a| a.floor_div(Division::Mod, 12, 13, 2));
        assert_eq!(lines.first().map(String::as_str), Some("push rdx"));
        assert!(lines.iter().any(|line| line == "idiv r11"), "{lines:?}");
        assert_eq!(lines.last().map(String::as_str), Some("pop rdx"));
    }

    #[test]
    fn unsigned_remainder_keeps_rdx() {
        let lines = listing(|a| a.unsigned_rem(12, 13, 2));
        assert_eq!(lines.first().map(String::as_str), Some("push rdx"));
        assert!(lines.iter().any(|line| line == "div r11"), "{lines:?}");
        assert_eq!(lines.last().map(String::as_str), Some("pop rdx"));
        let into_rdx = listing(|a| a.unsigned_rem(2, 13, 12));
        assert_eq!(into_rdx.last().map(String::as_str), Some("mov rdx,r11"), "{into_rdx:?}");
    }

    #[test]
    fn float_comparisons_test_parity_for_unordered_operands() {
        let lines = listing(|a| a.fset(FCond::Eq, 12, 1, 2));
        assert!(lines.contains(&"setnp r11b".to_string()), "{lines:?}");
        let lines = listing(|a| {
            let target = a.new_label();
            a.fbranch_not(FCond::Eq, 1, 2, target);
            a.bind(target);
        });
        assert!(lines.iter().any(|line| line.starts_with("jp ")), "{lines:?}");
    }

    #[test]
    fn select_and_set_use_conditional_moves() {
        assert_eq!(listing(|a| a.select(Cond::Lt, 12, 13, 14)), ["mov r12,r14", "cmovl r12,r13"]);
        assert_eq!(listing(|a| a.set(Cond::AboveEq, 6)), ["setae sil", "movzx esi,sil"]);
        assert_eq!(listing(|a| a.cmp_byte(15, 5)), ["cmp r15b,5"]);
    }

    #[test]
    fn memory_forms_address_the_stack_and_any_base() {
        assert_eq!(listing(|a| a.load(Width::B, 1, 4, 16)), ["movzx ecx,byte ptr [rsp+10h]"]);
        assert_eq!(listing(|a| a.load_i32(1, 3, 88)), ["movsxd rcx,[rbx+58h]"]);
        assert_eq!(listing(|a| a.store_imm(Width::Q, 12, 8, 1 << 40)), ["mov r11,10000000000h", "mov [r12+8],r11"]);
        assert_eq!(listing(|a| a.load_f(3, 13, -16)), ["movsd xmm3,[r13-10h]"]);
    }

    fn decoded(build: impl FnOnce(&mut X64)) -> Vec<Instruction> {
        let mut asm = X64::new();
        build(&mut asm);
        let code = asm.finish().expect("the code assembles");
        let mut decoder = Decoder::with_ip(64, &code, 0, DecoderOptions::NONE);
        let instructions: Vec<Instruction> = (&mut decoder).into_iter().collect();
        assert!(instructions.iter().all(|i| !i.is_invalid()), "{code:02x?}");
        instructions
    }

    fn pooled(build: impl FnOnce(&mut X64)) -> Instruction {
        let mut asm = X64::new();
        build(&mut asm);
        let code = asm.finish().expect("the code assembles");
        Decoder::with_ip(64, &code, 0, DecoderOptions::NONE).decode()
    }

    fn one(build: impl FnOnce(&mut X64)) -> Instruction {
        let instructions = decoded(build);
        assert_eq!(instructions.len(), 1, "{instructions:?}");
        instructions[0]
    }

    fn gpq(n: u8) -> Register {
        [
            Register::RAX,
            Register::RCX,
            Register::RDX,
            Register::RBX,
            Register::RSP,
            Register::RBP,
            Register::RSI,
            Register::RDI,
            Register::R8,
            Register::R9,
            Register::R10,
            Register::R11,
            Register::R12,
            Register::R13,
            Register::R14,
            Register::R15,
        ][n as usize]
    }

    fn gpd(n: u8) -> Register {
        [
            Register::EAX,
            Register::ECX,
            Register::EDX,
            Register::EBX,
            Register::ESP,
            Register::EBP,
            Register::ESI,
            Register::EDI,
            Register::R8D,
            Register::R9D,
            Register::R10D,
            Register::R11D,
            Register::R12D,
            Register::R13D,
            Register::R14D,
            Register::R15D,
        ][n as usize]
    }

    fn gpb(n: u8) -> Register {
        [
            Register::AL,
            Register::CL,
            Register::DL,
            Register::BL,
            Register::SPL,
            Register::BPL,
            Register::SIL,
            Register::DIL,
            Register::R8L,
            Register::R9L,
            Register::R10L,
            Register::R11L,
            Register::R12L,
            Register::R13L,
            Register::R14L,
            Register::R15L,
        ][n as usize]
    }

    fn xmm(n: u8) -> Register {
        [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
            Register::XMM8,
            Register::XMM9,
            Register::XMM10,
            Register::XMM11,
            Register::XMM12,
            Register::XMM13,
            Register::XMM14,
            Register::XMM15,
        ][n as usize]
    }

    const DISPS: [i32; 6] = [0, 8, -16, 127, -129, 0x1234];

    fn at(i: &Instruction, base: u8, disp: i32) {
        let found = (i.memory_base(), i.memory_index(), i.memory_displacement64() as i64);
        assert_eq!(found, (gpq(base), Register::None, i64::from(disp)), "{i}");
    }

    fn regs(i: &Instruction) -> Vec<Register> {
        (0..i.op_count()).filter(|&n| i.op_kind(n) == OpKind::Register).map(|n| i.op_register(n)).collect()
    }

    fn register_forms(a: u8, b: u8) {
        let i = one(|x| x.cmp(a, b));
        assert_eq!((i.mnemonic(), regs(&i)), (Mnemonic::Cmp, vec![gpq(a), gpq(b)]));
        for op in [Alu::Add, Alu::Sub, Alu::Mul, Alu::And, Alu::Or, Alu::Xor] {
            let i = one(|x| x.alu(op, a, a, b));
            assert_eq!(regs(&i), vec![gpq(a), gpq(b)], "{op:?} {a},{b}: {i}");
        }
        for op in [FAlu::Add, FAlu::Sub, FAlu::Mul, FAlu::Div] {
            let i = one(|x| x.falu(op, a, a, b));
            assert_eq!(regs(&i), vec![xmm(a), xmm(b)], "{op:?} {a},{b}");
        }
        assert_eq!(regs(&one(|x| x.bits_to_float(a, b))), vec![xmm(a), gpq(b)]);
        assert_eq!(regs(&one(|x| x.float_to_bits(a, b))), vec![gpq(a), xmm(b)]);
        assert_eq!(regs(&one(|x| x.ffloor(a, b))), vec![xmm(a), xmm(b)]);
        assert_eq!(regs(&one(|x| x.fsqrt(a, b))), vec![xmm(a), xmm(b)]);
        let i = pooled(|x| x.falu_const(FAlu::Mul, a, a, (1.5f64.to_bits(), b)));
        assert_eq!((i.mnemonic(), regs(&i), i.is_ip_rel_memory_operand()), (Mnemonic::Mulsd, vec![xmm(a)], true));
        assert_eq!(regs(&one(|x| x.zext32(a, b))), vec![gpd(a), gpd(b)]);
        assert_eq!(regs(&decoded(|x| x.int_to_float(a, b))[1]), vec![xmm(a), gpq(b)]);
        if a != b {
            assert_eq!(regs(&one(|x| x.mov(a, b))), vec![gpq(a), gpq(b)]);
            assert_eq!(regs(&one(|x| x.fmov(a, b))), vec![xmm(a), xmm(b)]);
        }
    }

    type Build<'a> = &'a dyn Fn(&mut X64);

    fn memory_forms(a: u8, b: u8, disp: i32) {
        let cases: [(Build<'_>, Mnemonic, Register); 10] = [
            (&|x| x.load(Width::Q, a, b, disp), Mnemonic::Mov, gpq(a)),
            (&|x| x.load(Width::W, a, b, disp), Mnemonic::Mov, gpd(a)),
            (&|x| x.load(Width::B, a, b, disp), Mnemonic::Movzx, gpd(a)),
            (&|x| x.store(Width::Q, a, b, disp), Mnemonic::Mov, gpq(a)),
            (&|x| x.store(Width::W, a, b, disp), Mnemonic::Mov, gpd(a)),
            (&|x| x.store(Width::B, a, b, disp), Mnemonic::Mov, gpb(a)),
            (&|x| x.lea(a, b, disp), Mnemonic::Lea, gpq(a)),
            (&|x| x.load_i32(a, b, disp), Mnemonic::Movsxd, gpq(a)),
            (&|x| x.load_f(a, b, disp), Mnemonic::Movsd, xmm(a)),
            (&|x| x.store_f(a, b, disp), Mnemonic::Movsd, xmm(a)),
        ];
        for (build, mnemonic, reg) in cases {
            let i = one(build);
            assert_eq!((i.mnemonic(), regs(&i)), (mnemonic, vec![reg]), "{i}");
            at(&i, b, disp);
        }
    }

    #[test]
    fn every_register_and_base_encodes_as_named() {
        for a in 0..16u8 {
            for b in 0..16u8 {
                register_forms(a, b);
                for disp in DISPS {
                    memory_forms(a, b, disp);
                }
            }
        }
    }

    fn immediate_forms(a: u8, value: i32) {
        let alu = [
            (Alu::Add, Mnemonic::Add),
            (Alu::Sub, Mnemonic::Sub),
            (Alu::And, Mnemonic::And),
            (Alu::Or, Mnemonic::Or),
            (Alu::Xor, Mnemonic::Xor),
            (Alu::Mul, Mnemonic::Imul),
        ];
        for (op, mnemonic) in alu {
            let i = one(|x| x.alu_imm(op, a, a, value));
            let found = (i.mnemonic(), i.op0_register(), i.immediate(i.op_count() - 1) as i64);
            assert_eq!(found, (mnemonic, gpq(a), i64::from(value)), "{i}");
        }
        let i = one(|x| x.cmp_imm(a, value));
        assert_eq!((i.mnemonic(), i.op0_register(), i.immediate(1) as i64), (Mnemonic::Cmp, gpq(a), i64::from(value)));
        let i = one(|x| x.test_imm(a, value));
        assert_eq!((i.mnemonic(), i.op0_register(), i.immediate(1) as i64), (Mnemonic::Test, gpq(a), i64::from(value)));
        let i = one(|x| x.store_imm(Width::Q, a, 8, i64::from(value)));
        at(&i, a, 8);
        assert_eq!(i.immediate(1) as i64, i64::from(value));
    }

    #[test]
    fn immediates_encode_their_value() {
        for a in 0..16u8 {
            for value in [0i64, 1, 0x7f, 0x80, 0xffff_ffff, -1, -0x8000_0000, 1 << 40, i64::MIN] {
                let i = one(|x| x.mov_imm(a, value));
                let mask = if i.op0_register() == gpd(a) { 0xffff_ffff } else { -1 };
                assert!(i.op0_register() == gpq(a) || i.op0_register() == gpd(a), "{i}");
                assert_eq!(i.immediate(1) as i64 & mask, value, "{i}");
            }
            for value in [0i32, 5, -1, 127, 128, -129, 0x12345] {
                immediate_forms(a, value);
            }
            let i = one(|x| x.cmp_byte(a, 0x93));
            assert_eq!((i.op0_register(), i.immediate(1)), (gpb(a), 0x93));
            let i = one(|x| x.store_imm(Width::B, a, -3, 0x93));
            at(&i, a, -3);
            assert_eq!(i.immediate(1), 0x93);
            let i = one(|x| x.store_imm(Width::W, a, 300, 7));
            at(&i, a, 300);
            assert_eq!(i.immediate(1), 7);
            let set = decoded(|x| x.set(Cond::Below, a));
            assert_eq!((set[0].mnemonic(), set[0].op0_register()), (Mnemonic::Setb, gpb(a)));
            assert_eq!(regs(&set[1]), vec![gpd(a), gpb(a)]);
            assert_eq!(one(|x| x.call_reg(a)).op0_register(), gpq(a));
            assert_eq!(regs(&one(|x| x.neg(a, a))), vec![gpq(a)]);
            assert_eq!(regs(&one(|x| x.not(a, a))), vec![gpq(a)]);
            let i = one(|x| x.shl_imm(a, a, 4));
            assert_eq!((i.mnemonic(), i.op0_register(), i.immediate(1)), (Mnemonic::Shl, gpq(a), 4));
        }
    }

    #[test]
    fn the_budget_is_spent_in_memory() {
        let i = one(|x| x.cmp_mem_byte(0, 0, 0));
        assert_eq!((i.mnemonic(), i.memory_size()), (Mnemonic::Cmp, iced_x86::MemorySize::UInt8));
        let i = one(|x| x.add_mem(13, 3, 0x18));
        assert_eq!((i.mnemonic(), i.op0_register()), (Mnemonic::Add, Register::R13));
        at(&i, 3, 0x18);
        let i = one(|x| x.sub_mem(12, 0, 300));
        assert_eq!((i.mnemonic(), i.immediate(1)), (Mnemonic::Sub, 300));
        at(&i, 12, 0);
    }

    #[test]
    fn branches_reach_their_labels() {
        let instructions = decoded(|x| {
            let back = x.new_label();
            let ahead = x.new_label();
            x.bind(back);
            x.branch(Cond::Le, ahead);
            x.jump(back);
            x.trap();
            x.bind(ahead);
        });
        assert_eq!(instructions[0].mnemonic(), Mnemonic::Jle);
        assert_eq!(instructions[0].near_branch_target(), 13);
        assert_eq!(instructions[1].near_branch_target(), 0);
    }

    #[test]
    fn the_raw_jump_lands_where_it_says() {
        let bytes = X64::raw_jump(0x40);
        let mut decoder = Decoder::with_ip(64, &bytes, 0, DecoderOptions::NONE);
        let instruction = decoder.decode();
        assert_eq!(instruction.near_branch_target(), 0x45);
    }
}

mod a64_encoding {
    use yaxpeax_arch::{Decoder, U8Reader};
    use yaxpeax_arm::armv8::a64::InstDecoder;

    use super::*;
    use crate::jit::ssa::emit::a64::A64;

    fn listing(build: impl FnOnce(&mut A64)) -> Vec<String> {
        let mut asm = A64::new();
        build(&mut asm);
        let code = asm.finish().expect("the code assembles");
        words(&code)
    }

    fn words(code: &[u8]) -> Vec<String> {
        assert_eq!(code.len() % 4, 0, "whole instructions");
        code.chunks(4)
            .map(|word| {
                let mut reader = U8Reader::new(word);
                let instruction = InstDecoder::default().decode(&mut reader).unwrap_or_else(|_| panic!("{word:02x?} decodes"));
                instruction.to_string()
            })
            .collect()
    }

    #[test]
    fn integer_forms() {
        assert_eq!(listing(|a| a.alu(Alu::Sub, 2, 1, 2)), ["sub x2, x1, x2"]);
        assert_eq!(listing(|a| a.alu_imm(Alu::Add, 20, 20, 4095)), ["add x20, x20, #0xfff"]);
        assert_eq!(listing(|a| a.alu_imm(Alu::Add, 20, 20, -8)), ["sub x20, x20, #0x8"]);
        assert_eq!(listing(|a| a.alu_imm(Alu::Add, 20, 21, 0x12345)), ["mov x17, #0x2345", "movk x17, #0x1, lsl #16", "add x20, x21, x17"]);
        assert_eq!(listing(|a| a.alu_imm(Alu::Xor, 9, 9, 1)), ["mov x17, #0x1", "eor x9, x9, x17"]);
        assert_eq!(listing(|a| a.shl_imm(3, 4, 4)), ["lsl x3, x4, #0x4"]);
        assert_eq!(listing(|a| a.zext32(3, 4)), ["mov w3, w4"]);
        assert_eq!(listing(|a| a.neg(3, 4)), ["neg x3, x4"]);
        assert_eq!(listing(|a| a.not(3, 4)), ["mvn x3, x4"]);
        assert_eq!(listing(|a| a.sub_mem(20, 0, 300)), ["ldr x17, [x20]", "subs x17, x17, #0x12c", "str x17, [x20]"]);
        assert_eq!(listing(|a| a.sub_mem(20, 0, 5000))[2], "subs x17, x17, x30");
        assert_eq!(listing(|a| a.add_mem(20, 19, 24)), ["ldr x17, [x19, #0x18]", "add x20, x20, x17"]);
    }

    #[test]
    fn comparisons_and_conditions() {
        assert_eq!(listing(|a| a.cmp_imm(5, 7)), ["cmp x5, #0x7"]);
        assert_eq!(listing(|a| a.cmp_imm(5, -1)), ["cmn x5, #0x1"]);
        assert_eq!(listing(|a| a.cmp_imm64(5, 1 << 40)), ["mov x17, #0x10000000000", "cmp x5, x17"]);
        assert_eq!(listing(|a| a.cmp_byte(5, 2)), ["uxtb w17, w5", "cmp w17, #0x2"]);
        assert_eq!(listing(|a| a.set(Cond::Lt, 3)), ["cset x3, lt"]);
        assert_eq!(listing(|a| a.set(Cond::AboveEq, 3)), ["cset x3, hs"]);
        assert_eq!(listing(|a| a.select(Cond::Ne, 3, 4, 5)), ["csel x3, x4, x5, ne"]);
    }

    #[test]
    fn memory_forms() {
        assert_eq!(listing(|a| a.load(Width::B, 1, 31, 16)), ["ldrb w1, [sp, #0x10]"]);
        assert_eq!(listing(|a| a.load(Width::Q, 1, 19, 8)), ["ldr x1, [x19, #0x8]"]);
        assert_eq!(listing(|a| a.load(Width::Q, 1, 19, -16)), ["ldur x1, [x19, #-0x10]"]);
        assert_eq!(listing(|a| a.load_i32(1, 19, 88)), ["ldrsw x1, [x19, #0x58]"]);
        assert_eq!(listing(|a| a.store(Width::W, 1, 2, 4)), ["str w1, [x2, #0x4]"]);
        assert_eq!(listing(|a| a.store_imm(Width::B, 2, 0, 0)), ["mov x30, #0x0", "strb w30, [x2]"]);
        assert_eq!(listing(|a| a.load(Width::Q, 1, 2, 40000)), ["mov x17, #0x9c40", "ldr x1, [x2, x17]"]);
        assert_eq!(listing(|a| a.load_f(3, 19, 8)), ["ldr d3, [x19, #0x8]"]);
        assert_eq!(listing(|a| a.lea(3, 31, 48)), ["add x3, sp, #0x30"]);
    }

    #[test]
    fn float_forms() {
        assert_eq!(listing(|a| a.falu(FAlu::Div, 0, 1, 0)), ["fdiv d0, d1, d0"]);
        assert_eq!(listing(|a| a.fneg(0, 1)), ["fneg d0, d1"]);
        assert_eq!(listing(|a| a.fsqrt(0, 1)), ["fsqrt d0, d1"]);
        assert_eq!(listing(|a| a.falu_const(FAlu::Sub, 0, 1, (0, 2))).last().map(String::as_str), Some("fsub d0, d1, d2"));
        assert!(listing(|a| a.ffloor(0, 1))[0].starts_with("frintm"));
        assert_eq!(listing(|a| a.bits_to_float(0, 9)), ["fmov d0, x9"]);
        assert_eq!(listing(|a| a.float_to_bits(9, 0)), ["fmov x9, d0"]);
        assert_eq!(listing(|a| a.int_to_float(0, 9)), ["scvtf d0, x9"]);
        assert_eq!(listing(|a| a.fset(FCond::Lt, 3, 0, 1)), ["fcmp d0, d1", "cset x3, mi"]);
        assert_eq!(listing(|a| a.fset(FCond::Le, 3, 0, 1)), ["fcmp d0, d1", "cset x3, ls"]);
    }

    #[test]
    fn unsigned_remainder_divides_and_multiplies_back() {
        assert_eq!(listing(|a| a.unsigned_rem(3, 4, 5)), ["udiv x17, x4, x5", "msub x3, x17, x5, x4"]);
    }

    #[test]
    fn floor_division_corrects_truncation() {
        let lines = listing(|a| a.floor_div(Division::IDiv, 3, 4, 5));
        assert_eq!(lines[0], "sdiv x17, x4, x5");
        assert_eq!(lines[1], "msub x30, x17, x5, x4");
        assert!(lines[2].starts_with("cbz x30"));
        assert_eq!(lines[3], "eor x16, x4, x5");
        assert!(lines[4].starts_with("tbz x16, #0x3f"));
        assert_eq!(lines[5], "sub x17, x17, #0x1");
        assert_eq!(lines[6], "mov x3, x17");
    }

    #[test]
    fn the_prologue_and_epilogue_mirror_each_other() {
        let mut asm = A64::new();
        let prologue = asm.prologue(64, &[], 16);
        asm.epilogue(64, &[], 3);
        let lines = words(&asm.finish().expect("the code assembles"));
        assert_eq!(lines[0], "stp x29, x30, [sp, #-0x10]!");
        assert_eq!(lines[1], "mov x29, sp");
        assert_eq!(lines[2], "stp x19, x20, [sp, #-0x10]!");
        assert_eq!(lines[6], "stp x27, x28, [sp, #-0x10]!");
        assert_eq!(lines[7], "sub sp, sp, #0x40");
        assert_eq!(lines[8], "mov x19, x0");
        assert_eq!(lines[9], "mov w16, w1");
        assert_eq!(lines[10], "mov x0, x3");
        assert_eq!(lines[11], "sub sp, x29, #0x50");
        assert_eq!(lines[12], "ldp x27, x28, [sp], #0x10");
        assert_eq!(lines[17], "ldp x29, x30, [sp], #0x10");
        assert_eq!(lines[18], "ret");
        assert_eq!(prologue.saves.len(), 5);
        assert_eq!(prologue.alloc, 32);
    }

    #[test]
    fn the_raw_jump_lands_where_it_says() {
        let lines = words(&A64::raw_jump(0x40));
        assert_eq!(lines, ["b $+0x44"]);
    }

    #[test]
    fn the_dump_stores_every_allocatable_register_at_its_index() {
        let lines = listing(|a| a.dump_registers());
        assert_eq!(lines[0], "sub sp, sp, #0x200");
        assert!(lines.contains(&"str x20, [sp, #0xa0]".to_string()), "{lines:?}");
        assert!(lines.contains(&"str d16, [sp, #0x180]".to_string()), "{lines:?}");
        assert_eq!(lines.len(), 1 + 25 + 23);
    }
}
