use crate::autosort::prime_factor::PrimeFactorFft32;
use crate::float::FftFloat;
use crate::Fft;
use num_complex::Complex;

fn compute_half_twiddle<T: FftFloat>(index: f64, size: usize) -> Complex<T> {
    let theta = index * std::f64::consts::PI / size as f64;
    let twiddle = Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    );
    twiddle
}

struct BluesteinsAlgorithm<T> {
    fft: Box<dyn Fft<Float = T>>,
    size: usize,
    w_forward: Box<[Complex<T>]>,
    w_inverse: Box<[Complex<T>]>,
    x_forward: Box<[Complex<T>]>,
    x_inverse: Box<[Complex<T>]>,
    work: Box<[Complex<T>]>,
}

impl<T: FftFloat> BluesteinsAlgorithm<T> {
    fn new<F: Fn(usize) -> Box<dyn Fft<Float = T>>>(size: usize, fft_maker: F) -> Self {
        let mut fft = fft_maker((2 * size - 1).checked_next_power_of_two().unwrap());

        // create W vector
        let mut w_forward = vec![Complex::default(); fft.size()].into_boxed_slice();
        let mut w_inverse = vec![Complex::default(); fft.size()].into_boxed_slice();
        for (i, (wfi, wii)) in w_forward.iter_mut().zip(w_inverse.iter_mut()).enumerate() {
            if let Some(index) = {
                if i < size {
                    Some((i as f64).powi(2))
                } else if i >= fft.size() - size + 1 {
                    Some(((i as f64) - (fft.size() as f64)).powi(2))
                } else {
                    None
                }
            } {
                *wfi = compute_half_twiddle(index, size);
                *wii = wfi.conj();
            }
        }
        fft.fft_in_place(&mut w_forward);
        fft.fft_in_place(&mut w_inverse);

        // create x vector
        let mut x_forward = vec![Complex::default(); size].into_boxed_slice();
        let mut x_inverse = vec![Complex::default(); size].into_boxed_slice();
        for (i, (xfi, xii)) in x_forward.iter_mut().zip(x_inverse.iter_mut()).enumerate() {
            *xfi = compute_half_twiddle(-(i as f64).powi(2), size);
            *xii = xfi.conj();
        }

        Self {
            work: vec![Complex::default(); fft.size()].into_boxed_slice(),
            fft,
            size,
            w_forward,
            w_inverse,
            x_forward,
            x_inverse,
        }
    }
}

impl<T: FftFloat> Fft for BluesteinsAlgorithm<T> {
    type Float = T;

    fn size(&self) -> usize {
        self.size
    }

    fn fft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {
        assert_eq!(input.len(), self.size);
        for (w, (x, i)) in self
            .work
            .iter_mut()
            .zip(self.x_forward.iter().zip(input.iter()))
        {
            *w = x * i;
        }
        for w in self.work[self.size..].iter_mut() {
            *w = Complex::default();
        }
        self.fft.fft_in_place(&mut self.work);
        for (w, wi) in self.work.iter_mut().zip(self.w_forward.iter()) {
            *w *= wi;
        }
        self.fft.ifft_in_place(&mut self.work);
        for (w, xi) in self.work.iter_mut().zip(self.x_forward.iter()) {
            *w *= xi;
        }
        input.copy_from_slice(&self.work[..self.size]);
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {}
}

pub fn create_f32(size: usize) -> Box<dyn Fft<Float = f32>> {
    Box::new(BluesteinsAlgorithm::new(size, |size| {
        Box::new(PrimeFactorFft32::new(size))
    }))
}
