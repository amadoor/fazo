//! Unwind tables for the Optimizing prologue, encoded as Cranelift's `isa/unwind` modules and `baseline/unwind.rs` encode them.

use cranelift_codegen::gimli::{
    self, Encoding, Format, LittleEndian, Register, X86_64,
    write::{Address, CallFrameInstruction, CommonInformationEntry, EhFrame, EndianVec, FrameDescriptionEntry, FrameTable},
};

use crate::jit::unwind::{Table, Trailer};

/// Code offsets are where each step ends.
#[derive(Clone, Debug, Default)]
pub(crate) struct Prologue {
    pub(crate) arch: Arch,
    pub(crate) save_frame: u32,
    pub(crate) set_frame: u32,
    /// x86-64: one register per push; AArch64: one pair per `stp`.
    pub(crate) saves: Vec<(u32, u8)>,
    pub(crate) alloc: u32,
    pub(crate) frame: u32,
    pub(crate) float_saves: Vec<(u32, u8, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Arch {
    #[default]
    X64,
    #[cfg(any(target_arch = "aarch64", test))]
    A64,
}

pub(crate) fn trailer(prologue: &Prologue, code_len: usize, offset: usize) -> Option<Trailer> {
    match prologue.arch {
        Arch::X64 if cfg!(windows) => Some(windows_x64(prologue, code_len, offset)),
        #[cfg(any(target_arch = "aarch64", test))]
        Arch::A64 if cfg!(windows) => windows_arm64(prologue, code_len, offset),
        _ => system_v(prologue, code_len, offset),
    }
}

/// UWOP_PUSH_NONVOL 0, UWOP_ALLOC_LARGE 1, UWOP_ALLOC_SMALL 2, UWOP_SAVE_XMM128 8, from the x64 exception handling
/// documentation (https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64), listed last step first.
fn windows_x64(prologue: &Prologue, code_len: usize, offset: usize) -> Trailer {
    let mut codes: Vec<[u8; 2]> = Vec::new();
    for &(end, register, stack) in prologue.float_saves.iter().rev() {
        codes.push(((stack / 16) as u16).to_le_bytes());
        codes.push([end as u8, (register << 4) | 8]);
    }
    let frame = prologue.frame;
    if frame > 0 && frame <= 128 {
        codes.push([prologue.alloc as u8, ((((frame - 8) / 8) as u8) << 4) | 2]);
    } else if frame > 0 {
        codes.push(((frame / 8) as u16).to_le_bytes());
        codes.push([prologue.alloc as u8, 1]);
    }
    for &(end, register) in prologue.saves.iter().rev() {
        codes.push([end as u8, register << 4]);
    }
    codes.push([prologue.save_frame as u8, 5 << 4]);
    let count = codes.len();
    let last = prologue.float_saves.last().map_or(prologue.alloc, |&(end, _, _)| end);
    let mut bytes = vec![1, last as u8, count as u8, 0];
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

/// Codes as in `baseline/unwind.rs` (https://learn.microsoft.com/en-us/cpp/build/arm64-exception-handling, "Unwind codes").
#[cfg(any(target_arch = "aarch64", test))]
fn windows_arm64(prologue: &Prologue, code_len: usize, offset: usize) -> Option<Trailer> {
    let frame = prologue.frame;
    let mut codes: Vec<u8> = Vec::new();
    let units = frame / 16;
    match units {
        0 => {}
        1..32 => codes.push(units as u8),
        32..2048 => codes.extend_from_slice(&[0xc0 | (units >> 8) as u8, units as u8]),
        _ => return None,
    }
    for pair in (0..prologue.saves.len()).rev() {
        let register = (2 * pair) as u8;
        codes.extend_from_slice(&[0xcc | (register >> 2), ((register & 3) << 6) | 1]);
    }
    codes.extend_from_slice(&[0xe1, 0x81, 0xe4]);
    codes.resize(codes.len().next_multiple_of(4), 0xe3);
    let code_words = (codes.len() / 4) as u32;
    let word = (code_len as u32 / 4) | (code_words << 27);
    let mut bytes = word.to_le_bytes().to_vec();
    bytes.extend_from_slice(&codes);
    let table = offset + bytes.len();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(offset as u32).to_le_bytes());
    Some(Trailer { bytes, table: Some(Table::WindowsArm64 { table }) })
}

fn system_v(prologue: &Prologue, code_len: usize, offset: usize) -> Option<Trailer> {
    let encoding = Encoding { address_size: 8, format: Format::Dwarf32, version: 1 };
    let start = Address::Constant((-(offset as i64)).cast_unsigned());
    let (mut cie, fde) = match prologue.arch {
        Arch::X64 => {
            let mut cie = CommonInformationEntry::new(encoding, 1, -8, X86_64::RA);
            cie.add_instruction(CallFrameInstruction::Cfa(X86_64::RSP, 8));
            cie.add_instruction(CallFrameInstruction::Offset(X86_64::RA, -8));
            let mut fde = FrameDescriptionEntry::new(start, code_len as u32);
            fde.add_instruction(prologue.save_frame, CallFrameInstruction::CfaOffset(16));
            fde.add_instruction(prologue.save_frame, CallFrameInstruction::Offset(X86_64::RBP, -16));
            fde.add_instruction(prologue.set_frame, CallFrameInstruction::CfaRegister(X86_64::RBP));
            for (index, &(end, register)) in prologue.saves.iter().enumerate() {
                let offset = -24 - 8 * index as i32;
                fde.add_instruction(end, CallFrameInstruction::Offset(Register(u16::from(dwarf_x64(register))), offset));
            }
            (cie, fde)
        }
        #[cfg(any(target_arch = "aarch64", test))]
        Arch::A64 => {
            use cranelift_codegen::gimli::AArch64;
            let mut cie = CommonInformationEntry::new(encoding, 4, -8, AArch64::X30);
            cie.add_instruction(CallFrameInstruction::Cfa(AArch64::SP, 0));
            let mut fde = FrameDescriptionEntry::new(start, code_len as u32);
            fde.add_instruction(prologue.save_frame, CallFrameInstruction::CfaOffset(16));
            fde.add_instruction(prologue.save_frame, CallFrameInstruction::Offset(AArch64::X29, -16));
            fde.add_instruction(prologue.save_frame, CallFrameInstruction::Offset(AArch64::X30, -8));
            fde.add_instruction(prologue.set_frame, CallFrameInstruction::CfaRegister(AArch64::X29));
            for (index, &(end, first)) in prologue.saves.iter().enumerate() {
                let base = -32 - 16 * index as i32;
                fde.add_instruction(end, CallFrameInstruction::Offset(Register(u16::from(first)), base));
                fde.add_instruction(end, CallFrameInstruction::Offset(Register(u16::from(first) + 1), base + 8));
            }
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

/// DWARF numbers rax, rdx, rcx, rbx, rsi, rdi, rbp, rsp as 0 to 7 (System V AMD64 ABI, figure 3.36).
fn dwarf_x64(register: u8) -> u8 {
    match register {
        1 => 2,
        2 => 1,
        3 => 3,
        6 => 4,
        7 => 5,
        other => other,
    }
}
