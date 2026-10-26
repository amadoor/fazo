#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Class {
    Gp,
    Fp,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PReg {
    pub(crate) class: Class,
    pub(crate) hw: u8,
}

impl PReg {
    pub(crate) const fn gp(hw: u8) -> Self {
        Self { class: Class::Gp, hw }
    }

    pub(crate) const fn fp(hw: u8) -> Self {
        Self { class: Class::Fp, hw }
    }

    pub(crate) fn index(self) -> usize {
        usize::from(self.hw) + if self.class == Class::Fp { 32 } else { 0 }
    }

    pub(crate) fn from_index(index: usize) -> Self {
        if index >= 32 { Self::fp((index - 32) as u8) } else { Self::gp(index as u8) }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct RegSet(pub(crate) u64);

impl RegSet {
    pub(crate) fn of(regs: &[PReg]) -> Self {
        regs.iter().fold(Self::default(), |set, &reg| set.with(reg))
    }

    pub(crate) fn with(self, reg: PReg) -> Self {
        Self(self.0 | 1 << reg.index())
    }

    pub(crate) fn insert(&mut self, reg: PReg) {
        self.0 |= 1 << reg.index();
    }

    pub(crate) fn contains(self, reg: PReg) -> bool {
        self.0 & 1 << reg.index() != 0
    }

    pub(crate) fn union(self, other: RegSet) -> Self {
        Self(self.0 | other.0)
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = PReg> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let index = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(PReg::from_index(index))
        })
    }
}

/// Allocation order, most preferred first.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Registers {
    pub(crate) gp: &'static [u8],
    pub(crate) fp: &'static [u8],
    pub(crate) callee_saved: RegSet,
}

impl Registers {
    pub(crate) fn order(&self, class: Class) -> impl Iterator<Item = PReg> + '_ {
        let (list, make): (&[u8], fn(u8) -> PReg) = match class {
            Class::Gp => (self.gp, PReg::gp),
            Class::Fp => (self.fp, PReg::fp),
        };
        list.iter().map(move |&hw| make(hw))
    }

    pub(crate) fn all(&self) -> RegSet {
        self.order(Class::Gp).chain(self.order(Class::Fp)).fold(RegSet::default(), RegSet::with)
    }
}
