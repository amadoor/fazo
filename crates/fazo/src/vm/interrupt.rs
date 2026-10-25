use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use crate::{LuaError, Value, Vm, session::INTERRUPT};

const CHECK_MASK: u32 = 0x3ff;

pub(crate) struct Interrupts {
    requests: Arc<AtomicU8>,
    ticks: u32,
}

impl Interrupts {
    pub(crate) fn new(vm: &Vm) -> Self {
        Self {
            requests: Arc::clone(&vm.requests),
            ticks: 0,
        }
    }

    #[inline]
    pub(crate) fn due(&mut self) -> bool {
        self.ticks = self.ticks.wrapping_add(1);
        self.ticks & CHECK_MASK == 0 && self.raised()
    }

    #[inline]
    pub(crate) fn raised(&self) -> bool {
        interrupt_requested(&self.requests)
    }

    #[cfg(test)]
    pub(crate) fn never() -> Self {
        Self {
            requests: Arc::new(AtomicU8::new(0)),
            ticks: 0,
        }
    }
}

#[inline]
pub(crate) fn interrupt_requested(requests: &AtomicU8) -> bool {
    requests.load(Ordering::Relaxed) & INTERRUPT != 0
}

impl Vm {
    pub(crate) fn check_interrupt(&mut self, interrupts: &mut Interrupts) -> Result<(), LuaError> {
        if interrupts.due() {
            return Err(self.interrupted());
        }
        Ok(())
    }

    pub(crate) fn check_raised(&mut self) -> Result<(), LuaError> {
        if interrupt_requested(&self.requests) {
            return Err(self.interrupted());
        }
        Ok(())
    }

    pub(crate) fn interrupted(&mut self) -> LuaError {
        self.requests.fetch_and(!INTERRUPT, Ordering::Relaxed);
        let message = self.heap.string(b"interrupted");
        LuaError::interrupted(Value::String(message))
    }
}
