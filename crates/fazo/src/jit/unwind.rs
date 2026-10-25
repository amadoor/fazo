//! Unwind data placed after a compiled function's code, and its OS registration. Ported from wasmtime (Apache-2.0 WITH LLVM-exception):
//! `src/runtime/vm/sys/unix/unwind.rs`, `src/runtime/vm/sys/windows/unwind64.rs` and `wasmtime-internal-cranelift/src/obj.rs`.
//! Offsets are relative to the blob start, so the same bytes work at any mapping address.

use std::io;

use cranelift_codegen::{
    gimli::{
        self, LittleEndian,
        write::{Address, EhFrame, EndianVec, FrameTable},
    },
    isa::{
        TargetIsa,
        unwind::{UnwindInfo, systemv, winarm64, winx64},
    },
};
use cranelift_module::ModuleError;

use super::{JitResult, sys};

pub(crate) const TRAILER_ALIGNMENT: usize = 16;

pub(crate) enum Table {
    WindowsX64 { table: usize },
    WindowsArm64 { table: usize },
    SystemV { eh_frame: usize, fdes: Vec<usize> },
}

pub(crate) struct Trailer {
    pub(crate) bytes: Vec<u8>,
    pub(crate) table: Option<Table>,
}

/// `offset` is where the trailer starts in the code blob; the result's offsets are from the blob start.
pub(super) fn build(
    isa: &dyn TargetIsa,
    info: Option<UnwindInfo>,
    code_len: usize,
    offset: usize,
) -> JitResult<Trailer> {
    let empty = Trailer {
        bytes: Vec::new(),
        table: None,
    };
    let Some(info) = info else {
        return Ok(empty);
    };
    match info {
        UnwindInfo::WindowsX64(info) => Ok(windows_x64(&info, code_len, offset)),
        UnwindInfo::WindowsArm64(info) => Ok(windows_arm64(&info, code_len, offset)),
        UnwindInfo::SystemV(info) => system_v(isa, &info, offset),
        _ => Err(unsupported("unwind information kind is not registered")),
    }
}

fn windows_x64(info: &winx64::UnwindInfo, code_len: usize, offset: usize) -> Trailer {
    let mut bytes = vec![0; info.emit_size()];
    info.emit(&mut bytes);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let table = offset + bytes.len();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(code_len).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(offset).unwrap().to_le_bytes());
    Trailer {
        bytes,
        table: Some(Table::WindowsX64 { table }),
    }
}

fn windows_arm64(info: &winarm64::UnwindInfo, code_len: usize, offset: usize) -> Trailer {
    let code_words = info.code_words();
    let mut unwind_codes = vec![0; usize::from(code_words) * 4];
    info.emit(&mut unwind_codes);

    let requires_extended_counts = code_words > (1 << 5);
    let encoded_function_len = u32::try_from(code_len / 4).unwrap();
    assert!(encoded_function_len < (1 << 18), "function too large");
    let mut word = encoded_function_len;
    if !requires_extended_counts {
        word |= u32::from(code_words) << 27;
    }

    let mut bytes = word.to_le_bytes().to_vec();
    if requires_extended_counts {
        bytes.extend_from_slice(&(u32::from(code_words) << 16).to_le_bytes());
    }
    bytes.extend_from_slice(&unwind_codes);
    bytes.resize(bytes.len().next_multiple_of(4), 0);

    let table = offset + bytes.len();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(offset).unwrap().to_le_bytes());
    Trailer {
        bytes,
        table: Some(Table::WindowsArm64 { table }),
    }
}

/// The FDE's pc-relative start is written so it resolves to the blob start; the zero terminator is what libgcc expects.
fn system_v(isa: &dyn TargetIsa, info: &systemv::UnwindInfo, offset: usize) -> JitResult<Trailer> {
    let Some(mut cie) = isa.create_systemv_cie() else {
        return Ok(Trailer {
            bytes: Vec::new(),
            table: None,
        });
    };
    cie.fde_address_encoding = gimli::constants::DW_EH_PE_pcrel;
    let mut frames = FrameTable::default();
    let cie_id = frames.add_cie(cie);
    let start = -i64::try_from(offset).unwrap();
    let fde = info.to_fde(Address::Constant(start.cast_unsigned()));
    frames.add_fde(cie_id, fde);

    let mut eh_frame = EhFrame(EndianVec::new(LittleEndian));
    frames
        .write_eh_frame(&mut eh_frame)
        .map_err(|_| unsupported("the eh_frame table could not be written"))?;
    let EhFrame(writer) = eh_frame;
    let mut bytes = writer.into_vec();
    let cie_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let fde_relative = 4 + cie_len;
    bytes.extend_from_slice(&0u32.to_le_bytes());

    Ok(Trailer {
        bytes,
        table: Some(Table::SystemV {
            eh_frame: offset,
            fdes: vec![offset + fde_relative],
        }),
    })
}

fn unsupported(message: &str) -> Box<ModuleError> {
    Box::new(ModuleError::Compilation(
        cranelift_codegen::CodegenError::Unsupported(message.into()),
    ))
}

fn allocation(err: io::Error) -> Box<ModuleError> {
    Box::new(ModuleError::Allocation { err })
}

/// # Safety
///
/// `code` must start the blob that `table` describes, and the blob must stay mapped until the registration is dropped.
pub(super) unsafe fn register(table: &Table, code: *const u8) -> JitResult<sys::UnwindRegistration> {
    debug_assert!(!code.is_null());
    match table {
        Table::WindowsX64 { table } | Table::WindowsArm64 { table } => {
            // SAFETY: the caller's contract puts the table inside the blob, and `code` is its base.
            let registration = unsafe {
                let entry = code.add(*table);
                debug_assert!(!entry.is_null());
                sys::register_function_table(entry, code)
            }
            .map_err(allocation)?;
            Ok(registration)
        }
        Table::SystemV { eh_frame, fdes } => {
            // SAFETY: the table's offsets lie inside the blob, so each computed pointer addresses mapped bytes.
            let registration = unsafe {
                let eh = code.add(*eh_frame);
                debug_assert!(!eh.is_null());
                let fdes: Vec<*const u8> = fdes.iter().map(|&fde| code.add(fde)).collect();
                sys::register_eh_frame(eh, &fdes)
            }
            .map_err(allocation)?;
            Ok(registration)
        }
    }
}
