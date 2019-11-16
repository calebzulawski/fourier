use crate::float::FftFloat;
use num_complex::Complex;

pub struct RadixConfig<T> {
    pub twiddles: Vec<Complex<T>>,
    pub stride: usize,
    pub size: usize,
}

impl<T: FftFloat> RadixConfig<T> {
    fn new(size: usize, stride: usize, radix: usize, forward: bool) -> Self {
        assert_eq!(size % radix, 0);
        let m = size / radix;
        let theta = T::from_i64(if forward { 2 } else { -2 }).unwrap() * T::PI()
            / T::from_usize(size).unwrap();
        let mut twiddles = Vec::new();
        for i in 0..m {
            let theta_i = T::from_usize(i).unwrap() * theta;
            twiddles.push(Complex::new(theta_i.cos(), -theta_i.sin()));
        }
        Self {
            twiddles,
            stride,
            size,
        }
    }

    pub fn forward(size: usize, stride: usize, radix: usize) -> Self {
        Self::new(size, stride, radix, true)
    }

    pub fn inverse(size: usize, stride: usize, radix: usize) -> Self {
        Self::new(size, stride, radix, false)
    }
}

#[inline]
fn radix2<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    RadixConfig {
        twiddles,
        stride,
        size,
    }: &RadixConfig<T>,
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

pub fn radix2_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &RadixConfig<f32>) {
    radix2(x, y, config);
}
