use std::sync::Arc;

use cranelift_codegen::{
    CodegenError, Context,
    control::ControlPlane,
    ir::Function,
    isa::OwnedTargetIsa,
    settings::{self, Configurable},
};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Module, ModuleError, ModuleReloc, default_libcall_names};

use super::{
    JitResult,
    memory::{CodeBudget, CodeMemory},
    sys, unwind,
};

/// Built once and reused, so a compile pays only for its function.
pub(crate) struct Compiler {
    isa: OwnedTargetIsa,
    context: Context,
    builder: FunctionBuilderContext,
}

impl Compiler {
    /// The optimizer and the backtracking allocator halve integer and float loop time against `opt_level = none` with the single-pass allocator.
    pub(crate) fn new() -> JitResult<Self> {
        let mut flags = settings::builder();
        let is_pic = if cfg!(target_arch = "x86_64") { "true" } else { "false" };
        let verifier = if cfg!(debug_assertions) { "true" } else { "false" };
        for (name, value) in [
            ("use_colocated_libcalls", "false"),
            ("is_pic", is_pic),
            ("opt_level", "speed"),
            ("enable_verifier", verifier),
            ("regalloc_algorithm", "backtracking"),
        ] {
            flags.set(name, value).map_err(|err| Box::new(ModuleError::Flag(err)))?;
        }
        let isa = cranelift_native::builder()
            .map_err(|message| Box::new(ModuleError::Compilation(CodegenError::Unsupported(message.to_owned()))))?
            .finish(settings::Flags::new(flags))
            .map_err(|err| Box::new(ModuleError::Compilation(err)))?;
        Ok(Self {
            isa,
            context: Context::new(),
            builder: FunctionBuilderContext::new(),
        })
    }
}

/// Dropping the unit unmaps the code, so its entry must not be called afterwards.
pub(crate) struct JitUnit {
    _unwind: Option<sys::UnwindRegistration>,
    _module: JITModule,
    entry: *const u8,
    #[cfg(test)]
    ir_insts: usize,
}

// SAFETY: the unit owns the code mapping and its module, and the entry pointer names only memory this unit maps.
unsafe impl Send for JitUnit {}

impl JitUnit {
    pub(crate) fn compile(
        compiler: &mut Compiler,
        budget: Arc<CodeBudget>,
        symbols: &[(&str, *const u8)],
        define: impl FnOnce(&mut JITModule, &mut Function, &mut FunctionBuilderContext) -> JitResult<FuncId>,
    ) -> JitResult<Self> {
        let mut builder = JITBuilder::with_isa(compiler.isa.clone(), default_libcall_names());
        builder.memory_provider(Box::new(CodeMemory::new(budget)));
        builder.symbols(symbols.iter().map(|&(name, address)| (name, address)));
        let mut module = JITModule::new(builder);

        let ctx = &mut compiler.context;
        ctx.clear();
        let id = define(&mut module, &mut ctx.func, &mut compiler.builder)?;
        #[cfg(test)]
        let ir_insts = ctx.func.dfg.num_insts();
        ctx.compile(module.isa(), &mut ControlPlane::default())
            .map_err(|err| Box::new(ModuleError::from(err)))?;
        let compiled = ctx.compiled_code().expect("the function was just compiled");
        let code = compiled.code_buffer();
        let alignment = u64::from(compiled.buffer.min_alignment)
            .max(module.isa().function_alignment().minimum as u64)
            .max(module.isa().symbol_alignment());
        let relocs: Vec<ModuleReloc> = compiled
            .buffer
            .relocs()
            .iter()
            .map(|reloc| ModuleReloc::from_mach_reloc(reloc, &ctx.func, id))
            .collect();
        let info = compiled
            .create_unwind_info(module.isa())
            .map_err(|err| Box::new(ModuleError::from(err)))?;

        let trailer_offset = code.len().next_multiple_of(unwind::TRAILER_ALIGNMENT);
        let trailer = unwind::build(module.isa(), info, code.len(), trailer_offset)?;
        let mut blob = Vec::with_capacity(trailer_offset + trailer.bytes.len());
        blob.extend_from_slice(code);
        blob.resize(trailer_offset, 0);
        blob.extend_from_slice(&trailer.bytes);

        module.define_function_bytes(id, alignment, &blob, &relocs)?;
        module.finalize_definitions()?;
        let entry = module.get_finalized_function(id);

        let unwind = match trailer.table {
            // SAFETY: `entry` starts the finalized blob `trailer` was built for, and the unit's module keeps it mapped until the registration drops.
            Some(table) => Some(unsafe { unwind::register(&table, entry) }?),
            None => None,
        };
        Ok(Self {
            _unwind: unwind,
            _module: module,
            entry,
            #[cfg(test)]
            ir_insts,
        })
    }

    #[cfg(test)]
    pub(crate) fn ir_insts(&self) -> usize {
        self.ir_insts
    }

    pub(crate) fn entry(&self) -> *const u8 {
        self.entry
    }
}
