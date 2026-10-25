pub(crate) const MAX_LMEM: i64 = i64::MAX;
pub(crate) const PARAM_COUNT: usize = 6;

const DEFAULT_STEPSIZE: u32 = 200 * 56;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Param {
    MinorMul,
    MajorMinor,
    MinorMajor,
    Pause,
    StepMul,
    StepSize,
}

impl Param {
    pub(crate) const ALL: [Param; PARAM_COUNT] = [
        Param::MinorMul,
        Param::MajorMinor,
        Param::MinorMajor,
        Param::Pause,
        Param::StepMul,
        Param::StepSize,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn default_percent(self) -> u32 {
        match self {
            Param::MinorMul => 20,
            Param::MajorMinor => 50,
            Param::MinorMajor => 70,
            Param::Pause => 250,
            Param::StepMul => 200,
            Param::StepSize => DEFAULT_STEPSIZE,
        }
    }
}

pub(crate) fn code_param(percent: u32) -> u8 {
    if percent >= (0x1F << 7) * 100 {
        return 0xFF;
    }
    let p = (percent * 128).div_ceil(100);
    if p < 0x10 {
        return p as u8;
    }
    let log = ceil_log2(p + 1) - 5;
    (((p >> log) - 0x10) | ((log + 1) << 4)) as u8
}

pub(crate) fn apply_param(code: u8, x: i64) -> i64 {
    let mut mantissa = i64::from(code & 0xF);
    let mut exponent = i64::from(code >> 4);
    if exponent > 0 {
        exponent -= 1;
        mantissa += 0x10;
    }
    exponent -= 7;
    if exponent >= 0 {
        return if x < (MAX_LMEM / 0x1F) >> exponent {
            (x * mantissa) << exponent
        } else {
            MAX_LMEM
        };
    }
    let shift = -exponent;
    if x < MAX_LMEM / 0x1F {
        (x * mantissa) >> shift
    } else if (x >> shift) < MAX_LMEM / 0x1F {
        (x >> shift) * mantissa
    } else {
        MAX_LMEM
    }
}

fn ceil_log2(x: u32) -> u32 {
    if x <= 1 { 0 } else { 32 - (x - 1).leading_zeros() }
}

#[cfg(test)]
mod tests {
    use super::{Param, apply_param, code_param};

    #[test]
    fn parameters_round_trip_through_their_byte() {
        for percent in [0, 2, 10, 90, 100, 250, 500, 5000] {
            assert_eq!(apply_param(code_param(percent), 100), percent as i64);
        }
        assert_eq!(apply_param(code_param(Param::Pause.default_percent()), 100), 250);
    }
}
