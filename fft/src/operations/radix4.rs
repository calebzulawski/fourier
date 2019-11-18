use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Radix4<T> {
    base: BaseConfig<T>,
    forward: bool,
}

impl<T: FftFloat> Radix4<T> {
    pub fn forward(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::forward(size, stride, 4),
            forward: true,
        }
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::inverse(size, stride, 4),
            forward: false,
        }
    }
}

#[inline]
pub fn radix4<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix4 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
        forward,
    }: &Radix4<T>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    fn rotate<T: FftFloat>(z: Complex<T>, inverse: bool) -> Complex<T> {
        if inverse {
            Complex::new(-z.im, z.re)
        } else {
            Complex::new(z.im, -z.re)
        }
    }

    if *size == 4usize {
        for i in 0..*stride {
            let x0 = x[i];
            let x1 = x[i + stride];
            let x2 = x[i + 2 * stride];
            let x3 = x[i + 3 * stride];
            let y0 = x0 + x2;
            let y1 = x0 - x2;
            let y2 = x1 + x3;
            let y3 = rotate(x1 - x3, *forward);
            y[i] = y0 + y2;
            y[i + stride] = y1 - y3;
            y[i + 2 * stride] = y0 - y2;
            y[i + 3 * stride] = y1 + y3;
        }
    } else {
        let m = size / 4;
        for i in 0..m {
            let wi1 = twiddles[i];
            let wi2 = twiddles[i + m];
            let wi3 = twiddles[i + 2 * m];
            for j in 0..*stride {
                let x0 = x[j + stride * i];
                let x1 = x[j + stride * (i + m)];
                let x2 = x[j + stride * (i + 2 * m)];
                let x3 = x[j + stride * (i + 3 * m)];
                let y0 = x0 + x2;
                let y1 = x0 - x2;
                let y2 = x1 + x3;
                let y3 = rotate(x1 - x3, *forward);
                y[j + stride * (4 * i + 0)] = y0 + y2;
                y[j + stride * (4 * i + 1)] = (y1 - y3) * wi1;
                y[j + stride * (4 * i + 2)] = (y0 - y2) * wi2;
                y[j + stride * (4 * i + 3)] = (y1 + y3) * wi3;
            }
        }
    }
}

pub fn radix4_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    radix4(x, y, config);
}
