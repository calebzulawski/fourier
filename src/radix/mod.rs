use crate::float::FftFloat;
use num_complex::Complex;

mod radix2;
mod radix3;
pub use radix2::*;
pub use radix3::*;

fn compute_twiddle<T: FftFloat>(index: usize, size: usize, forward: bool) -> Complex<T> {
    let theta = (index * 2) as f64 * std::f64::consts::PI / size as f64;
    let twiddle = Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    );
    if forward {
        twiddle
    } else {
        twiddle.conj()
    }
}

#[derive(Debug)]
struct BaseConfig<T> {
    twiddles: Vec<Complex<T>>,
    stride: usize,
    size: usize,
}

impl<T: FftFloat> BaseConfig<T> {
    fn new(size: usize, stride: usize, radix: usize, forward: bool) -> Self {
        assert_eq!(size % radix, 0);
        let m = size / radix;
        let mut twiddles = Vec::new();
        for i in 1..radix {
            for j in 0..m {
                twiddles.push(compute_twiddle(i * j, size, forward));
            }
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
