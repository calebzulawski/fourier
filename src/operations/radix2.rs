use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Radix2<T> {
    base: BaseConfig<T>,
}

impl<T: FftFloat> Radix2<T> {
    pub fn forward(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::forward(size, stride, 2),
        }
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::inverse(size, stride, 2),
        }
    }
}

#[inline]
pub fn radix2<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix2 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
    }: &Radix2<T>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    let m = size / 2;
    for i in 0..m {
        let wi = twiddles[i];
        for j in 0..*stride {
            let a = x[j + stride * i];
            let b = x[j + stride * (i + m)];
            y[j + stride * 2 * i] = a + b;
            y[j + stride * (2 * i + 1)] = (a - b) * wi;
        }
    }
}

pub fn radix2_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix2<f32>) {
    radix2(x, y, config);
}
