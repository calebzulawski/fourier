use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Radix3<T> {
    base: BaseConfig<T>,
    twiddle: Complex<T>,
}

impl<T: FftFloat> Radix3<T> {
    fn new(size: usize, stride: usize, forward: bool) -> Self {
        Self {
            base: BaseConfig::new(size, stride, 3, forward),
            twiddle: super::compute_twiddle(1, 3, forward),
        }
    }

    pub fn forward(size: usize, stride: usize) -> Self {
        Self::new(size, stride, true)
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self::new(size, stride, false)
    }
}

#[inline]
pub fn radix3<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix3 {
        base: config,
        twiddle,
    }: &Radix3<T>,
) {
    let bfly = |x: [Complex<T>; 3], _forward: bool| -> [Complex<T>; 3] {
        [
            x[0] + x[1] + x[2],
            x[0] + x[1] * twiddle + x[2] * twiddle.conj(),
            x[0] + x[1] * twiddle.conj() + x[2] * twiddle,
        ]
    };

    crate::implement_generic! {3, x, y, config, bfly}
}

pub fn radix3_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix3<f32>) {
    radix3(x, y, config);
}
