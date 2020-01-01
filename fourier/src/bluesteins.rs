use crate::float::FftFloat;
use crate::Fft;
use num_complex::Complex;

pub fn compute_half_twiddle<T: FftFloat>(index: f64, size: usize, forward: bool) -> Complex<T> {
    let theta = index * std::f64::consts::PI / size as f64;
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

struct BluesteinsAlgorithm<T> {
    fft: Box<dyn Fft<Float = T>>,
    size: usize,
    forward: bool,
    w: Box<[Complex<T>]>,
    x: Box<[Complex<T>]>,
}

impl<T: FftFloat> BluesteinsAlgorithm<T> {
    fn new<F: Fn(usize) -> Box<dyn Fft<Float = T>>>(
        size: usize,
        fft_maker: F,
        forward: bool,
    ) -> Self {
        let mut fft = fft_maker((2 * size - 1).checked_next_power_of_two().unwrap());

        // create W vector
        let mut w = vec![Complex::default(); fft.size()].into_boxed_slice();
        for (i, wi) in w.iter_mut().enumerate() {
            if let Some(index) = {
                if i < size {
                    Some((i as f64).powi(2))
                } else if i >= fft.size() - size + 1 {
                    Some(((i as f64) - (fft.size() as f64)).powi(2))
                } else {
                    None
                }
            } {
                *wi = compute_half_twiddle(index, size, forward);
            }
        }
        fft.fft_in_place(&mut w);

        // create x vector
        let mut x = vec![Complex::default(); size].into_boxed_slice();
        for (i, xi) in w.iter_mut().enumerate() {
            *xi = compute_half_twiddle((i as f64).powi(2), size, forward);
        }

        Self {
            fft,
            size,
            forward,
            w,
            x,
        }
    }
}

impl<T> Fft for BluesteinsAlgorithm<T> {
    type Float = T;

    fn size(&self) -> usize {
        self.size
    }

    fn fft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {}
    fn ifft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {}
}
