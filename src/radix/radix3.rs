use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

#[derive(Debug)]
pub struct Radix3Config<T> {
    base: BaseConfig<T>,
    twiddle: Complex<T>,
}

impl<T: FftFloat> Radix3Config<T> {
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
    Radix3Config {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
        twiddle,
    }: &Radix3Config<T>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    let m = size / 3;
    for i in 0..m {
        let wi1 = twiddles[i];
        let wi2 = twiddles[i + m];
        for j in 0..*stride {
            let a = x[j + stride * i];
            let b = x[j + stride * (i + m)];
            let c = x[j + stride * (i + 2 * m)];
            y[j + stride * 3 * i] = a + b + c;
            y[j + stride * (3 * i + 1)] = (a + b * twiddle + c * twiddle.conj()) * wi1;
            y[j + stride * (3 * i + 2)] = (a + b * twiddle.conj() + c * twiddle) * wi2;
        }
    }
}

pub fn radix3_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix3Config<f32>) {
    radix3(x, y, config);
}
