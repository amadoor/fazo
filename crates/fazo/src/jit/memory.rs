use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::{ModuleError, ModuleResult};

use super::sys;

pub(super) const GRANULE: usize = 64 * 1024;

pub(crate) struct CodeBudget {
    limit: usize,
    used: AtomicUsize,
}

impl CodeBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
        })
    }

    #[cfg(test)]
    pub(super) fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    pub(crate) fn reserve(&self, bytes: usize) -> io::Result<()> {
        let previous = self.used.fetch_add(bytes, Ordering::AcqRel);
        if previous + bytes > self.limit {
            self.used.fetch_sub(bytes, Ordering::AcqRel);
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "the code budget is exhausted",
            ));
        }
        Ok(())
    }

    pub(crate) fn release(&self, bytes: usize) {
        self.used.fetch_sub(bytes, Ordering::AcqRel);
    }
}

struct Region {
    start: *mut u8,
    len: usize,
    executable: bool,
}

// SAFETY: the mapping is owned exclusively by its `CodeMemory`, so moving the owner moves the only handle to it.
unsafe impl Send for Region {}

/// Executable memory for one compiled unit. Regions are writable until `finalize` makes them executable, never both.
pub(super) struct CodeMemory {
    budget: Arc<CodeBudget>,
    regions: Vec<Region>,
}

impl CodeMemory {
    pub(super) fn new(budget: Arc<CodeBudget>) -> Self {
        Self {
            budget,
            regions: Vec::new(),
        }
    }

    fn release_all(&mut self) {
        sys::end_write();
        for region in self.regions.drain(..) {
            debug_assert!(!region.start.is_null() && region.len > 0);
            // SAFETY: each region came from `sys::reserve` in `allocate`, and nothing else holds a pointer into it.
            unsafe { sys::release(region.start, region.len) };
            self.budget.release(region.len);
        }
    }
}

impl JITMemoryProvider for CodeMemory {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        if !matches!(kind, JITMemoryKind::Executable) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "only executable memory is provided",
            ));
        }
        if align > GRANULE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the alignment is larger than the allocation granule",
            ));
        }
        let len = size.max(1).next_multiple_of(GRANULE);
        self.budget.reserve(len)?;
        let start = match sys::reserve(len) {
            Ok(start) => start,
            Err(error) => {
                self.budget.release(len);
                return Err(error);
            }
        };
        self.regions.push(Region {
            start,
            len,
            executable: false,
        });
        Ok(start)
    }

    fn finalize(&mut self, _branch_protection: BranchProtection) -> ModuleResult<()> {
        for region in self.regions.iter_mut().filter(|region| !region.executable) {
            debug_assert!(!region.start.is_null() && region.len > 0);
            // SAFETY: `finalize` runs only after every definition of the unit is written, so the bytes are complete.
            unsafe { sys::make_executable(region.start, region.len) }
                .map_err(|err| ModuleError::Allocation { err })?;
            debug_assert!(!region.start.is_null() && region.len > 0);
            // SAFETY: the range is the region just made executable, so every byte of it is mapped.
            unsafe {
                wasmtime_jit_icache_coherence::clear_cache(region.start.cast(), region.len)
            }
            .map_err(|err| ModuleError::Allocation {
                err: io::Error::other(err.to_string()),
            })?;
            region.executable = true;
        }
        wasmtime_jit_icache_coherence::pipeline_flush_mt().map_err(|err| ModuleError::Allocation {
            err: io::Error::other(err.to_string()),
        })
    }

    unsafe fn free_memory(&mut self) {
        self.release_all();
    }
}

impl Drop for CodeMemory {
    fn drop(&mut self) {
        self.release_all();
    }
}
