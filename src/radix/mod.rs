use crate::float::FftFloat;
use num_complex::Complex;

mod radix2;
pub use radix2::*;

fn twiddle<T: FftFloat>(index: usize, size: usize, forward: bool) -> Complex<T> {
    let theta = T::from_usize(index * 2).unwrap() * T::PI() / T::from_usize(size).unwrap();
    let twiddle = Complex::new(theta.cos(), -theta.sin());
    if forward {
        twiddle
    } else {
        twiddle.conj()
    }
}

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
        for i in 0..m {
            twiddles.push(twiddle(i, m, forward));
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
