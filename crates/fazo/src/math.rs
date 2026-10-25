//! The platform math library matches PUC on each platform; the `libm` feature trades that for identical results everywhere.

#[cfg(not(feature = "libm"))]
mod imp {
    pub fn sin(x: f64) -> f64 {
        x.sin()
    }

    pub fn cos(x: f64) -> f64 {
        x.cos()
    }

    pub fn tan(x: f64) -> f64 {
        x.tan()
    }

    pub fn asin(x: f64) -> f64 {
        x.asin()
    }

    pub fn acos(x: f64) -> f64 {
        x.acos()
    }

    pub fn atan2(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }

    pub fn exp(x: f64) -> f64 {
        x.exp()
    }

    pub fn ln(x: f64) -> f64 {
        x.ln()
    }

    pub fn log2(x: f64) -> f64 {
        x.log2()
    }

    pub fn log10(x: f64) -> f64 {
        x.log10()
    }

    pub fn pow(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
}

#[cfg(feature = "libm")]
mod imp {
    pub fn sin(x: f64) -> f64 {
        libm::sin(x)
    }

    pub fn cos(x: f64) -> f64 {
        libm::cos(x)
    }

    pub fn tan(x: f64) -> f64 {
        libm::tan(x)
    }

    pub fn asin(x: f64) -> f64 {
        libm::asin(x)
    }

    pub fn acos(x: f64) -> f64 {
        libm::acos(x)
    }

    pub fn atan2(y: f64, x: f64) -> f64 {
        libm::atan2(y, x)
    }

    pub fn exp(x: f64) -> f64 {
        libm::exp(x)
    }

    pub fn ln(x: f64) -> f64 {
        libm::log(x)
    }

    pub fn log2(x: f64) -> f64 {
        libm::log2(x)
    }

    pub fn log10(x: f64) -> f64 {
        libm::log10(x)
    }

    pub fn pow(x: f64, y: f64) -> f64 {
        libm::pow(x, y)
    }
}

pub(crate) use imp::*;
