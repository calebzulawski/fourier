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

fn apply<T: FftFloat>(
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
    x: &[Complex<T>],
    w: &[Complex<T>],
    size: usize,
    fft: &mut Box<dyn Fft<Float = T>>,
    forward: bool,
) {
    assert_eq!(input.len(), size);
    for (w, (x, i)) in work.iter_mut().zip(x.iter().zip(input.iter())) {
        *w = x * i;
    }
    for w in work[size..].iter_mut() {
        *w = Complex::default();
    }
    fft.fft_in_place(work);
    for (w, wi) in work.iter_mut().zip(w.iter()) {
        *w *= wi;
    }
    fft.ifft_in_place(work);
    if forward {
        for (i, (w, xi)) in input.iter_mut().zip(work.iter().zip(x.iter())) {
            *i = w * xi;
        }
    } else {
        for (i, (w, xi)) in input.iter_mut().zip(work.iter().zip(x.iter())) {
            *i = w * xi / T::from_usize(size).unwrap();
        }
    }

    // TODO: this shouldn't be necessary...
    input[1..].reverse();
}

impl<T: FftFloat> Fft for BluesteinsAlgorithm<T> {
    type Float = T;

    fn size(&self) -> usize {
        self.size
    }

    fn fft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {
        apply(
            input,
            &mut self.work,
            &self.x_forward,
            &self.w_forward,
            self.size,
            &mut self.fft,
            true,
        );
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<Self::Float>]) {
        apply(
            input,
            &mut self.work,
            &self.x_inverse,
            &self.w_inverse,
            self.size,
            &mut self.fft,
            false,
        );
    }
}

pub fn create_f32(size: usize) -> Box<dyn Fft<Float = f32>> {
    Box::new(BluesteinsAlgorithm::new(size, |size| {
        crate::autosort::prime_factor::create_f32(size).unwrap()
    }))
}
