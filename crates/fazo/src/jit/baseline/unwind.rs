//! Unwind tables for the Baseline prologue, in the formats `jit/unwind.rs` registers, encoded as Cranelift's `isa/unwind` modules do.

use cranelift_codegen::gimli::{
    self, AArch64, Encoding, Format, LittleEndian, Register, X86_64,
    write::{Address, CallFrameInstruction, CommonInformationEntry, EhFrame, EndianVec, FrameDescriptionEntry, FrameTable},
};

use crate::jit::unwind::{Table, Trailer};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Prologue {
    #[cfg_attr(all(target_arch = "aarch64", not(test)), expect(dead_code, reason = "only x86-64 templates have it"))]
    X64 { push_rbp: u32, set_rbp: u32, pushes: [(u32, u8); 4], alloc: u32, reserved: u32 },
    #[cfg_attr(all(target_arch = "x86_64", not(test)), expect(dead_code, reason = "only AArch64 frames have it"))]
    A64 { save_frame: u32, set_frame: u32, pairs: [u32; 2], reserved: u32 },
}

pub(crate) fn trailer(prologue: &Prologue, code_len: usize, offset: usize) -> Option<Trailer> {
    match prologue {
        Prologue::X64 { .. } if cfg!(windows) => Some(windows_x64(prologue, code_len, offset)),
        Prologue::A64 { .. } if cfg!(windows) => Some(windows_arm64(prologue, code_len, offset)),
        _ => system_v(prologue, code_len, offset),
    }
}

fn windows_x64(prologue: &Prologue, code_len: usize, offset: usize) -> Trailer {
    let Prologue::X64 { push_rbp, pushes, alloc, reserved, .. } = *prologue else {
        unreachable!("an x86-64 prologue");
    };
    let mut codes: Vec<[u8; 2]> = Vec::new();
    if reserved <= 128 {
        codes.push([alloc as u8, ((((reserved - 8) / 8) as u8) << 4) | 2]);
    } else {
        let scaled = (reserved / 8) as u16;
        codes.push([alloc as u8, 1]);
        codes.push(scaled.to_le_bytes());
    }
    for &(end, register) in pushes.iter().rev() {
        codes.push([end as u8, register << 4]);
    }
    codes.push([push_rbp as u8, 5 << 4]);
    let count = codes.len();
    let mut bytes = vec![1, alloc as u8, count as u8, 0];
    for code in codes {
        bytes.extend_from_slice(&code);
    }
    if count % 2 == 1 {
        bytes.extend_from_slice(&[0, 0]);
    }
    let table = offset + bytes.len();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(code_len as u32).to_le_bytes());
    bytes.extend_from_slice(&(offset as u32).to_le_bytes());
    Trailer { bytes, table: Some(Table::WindowsX64 { table }) }
}

fn windows_arm64(prologue: &Prologue, code_len: usize, offset: usize) -> Trailer {
    let Prologue::A64 { reserved, .. } = *prologue else {
        unreachable!("an AArch64 prologue");
    };
    let mut codes: Vec<u8> = Vec::new();
    if reserved > 0 {
        debug_assert!(reserved / 16 < 32);
        codes.push((reserved / 16) as u8);
    }
    codes.extend_from_slice(&[0xcc, 0x81, 0xcc, 0x01, 0xe1, 0x81, 0xe4]);
    codes.resize(codes.len().next_multiple_of(4), 0xe3);
    let code_words = (codes.len() / 4) as u32;
    let word = (code_len as u32 / 4) | (code_words << 27);
    let mut bytes = word.to_le_bytes().to_vec();
    bytes.extend_from_slice(&codes);
    let table = offset + bytes.len();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(offset as u32).to_le_bytes());
    Trailer { bytes, table: Some(Table::WindowsArm64 { table }) }
}

fn system_v(prologue: &Prologue, code_len: usize, offset: usize) -> Option<Trailer> {
    let encoding = Encoding { address_size: 8, format: Format::Dwarf32, version: 1 };
    let start = Address::Constant((-(offset as i64)).cast_unsigned());
    let (mut cie, fde) = match *prologue {
        Prologue::X64 { push_rbp, set_rbp, pushes, .. } => {
            let mut cie = CommonInformationEntry::new(encoding, 1, -8, X86_64::RA);
            cie.add_instruction(CallFrameInstruction::Cfa(X86_64::RSP, 8));
            cie.add_instruction(CallFrameInstruction::Offset(X86_64::RA, -8));
            let mut fde = FrameDescriptionEntry::new(start, code_len as u32);
            fde.add_instruction(push_rbp, CallFrameInstruction::CfaOffset(16));
            fde.add_instruction(push_rbp, CallFrameInstruction::Offset(X86_64::RBP, -16));
            fde.add_instruction(set_rbp, CallFrameInstruction::CfaRegister(X86_64::RBP));
            for (index, &(end, register)) in pushes.iter().enumerate() {
                let offset = -24 - 8 * index as i32;
                fde.add_instruction(end, CallFrameInstruction::Offset(Register(u16::from(dwarf_x64(register))), offset));
            }
            (cie, fde)
        }
        Prologue::A64 { save_frame, set_frame, pairs, .. } => {
            let mut cie = CommonInformationEntry::new(encoding, 4, -8, AArch64::X30);
            cie.add_instruction(CallFrameInstruction::Cfa(AArch64::SP, 0));
            let mut fde = FrameDescriptionEntry::new(start, code_len as u32);
            fde.add_instruction(save_frame, CallFrameInstruction::CfaOffset(16));
            fde.add_instruction(save_frame, CallFrameInstruction::Offset(AArch64::X29, -16));
            fde.add_instruction(save_frame, CallFrameInstruction::Offset(AArch64::X30, -8));
            fde.add_instruction(set_frame, CallFrameInstruction::CfaRegister(AArch64::X29));
            fde.add_instruction(pairs[0], CallFrameInstruction::Offset(AArch64::X19, -32));
            fde.add_instruction(pairs[0], CallFrameInstruction::Offset(AArch64::X20, -24));
            fde.add_instruction(pairs[1], CallFrameInstruction::Offset(AArch64::X21, -48));
            fde.add_instruction(pairs[1], CallFrameInstruction::Offset(AArch64::X22, -40));
            (cie, fde)
        }
    };
    cie.fde_address_encoding = gimli::constants::DW_EH_PE_pcrel;
    let mut frames = FrameTable::default();
    let cie_id = frames.add_cie(cie);
    frames.add_fde(cie_id, fde);
    let mut eh_frame = EhFrame(EndianVec::new(LittleEndian));
    frames.write_eh_frame(&mut eh_frame).ok()?;
    let EhFrame(writer) = eh_frame;
    let mut bytes = writer.into_vec();
    let cie_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    bytes.extend_from_slice(&0u32.to_le_bytes());
    Some(Trailer {
        bytes,
        table: Some(Table::SystemV { eh_frame: offset, fdes: vec![offset + 4 + cie_len] }),
    })
}

fn dwarf_x64(register: u8) -> u8 {
    match register {
        3 => X86_64::RBX.0 as u8,
        other => other,
    }
}
