use crate::fft::{Fft, Transform};
use crate::float::FftFloat;
use num_complex::Complex;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt, powi without std

fn compute_half_twiddle<T: FftFloat>(index: f64, size: usize) -> Complex<T> {
    let theta = index * core::f64::consts::PI / size as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}

pub struct Bluesteins {
    size: usize,
    inner_size: usize,
}

impl Bluesteins {
    pub fn new(size: usize) -> Self {
        Self {
            size: size,
            inner_size: (2 * size - 1).checked_next_power_of_two().unwrap(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn inner_fft_size(&self) -> usize {
        self.inner_size
    }

    pub fn initialize_w_twiddles<
        T: FftFloat,
        E: Extend<Complex<T>> + AsMut<[Complex<T>]>,
        F: Fft<Real = T>,
    >(
        &self,
        fft: &F,
        twiddles: &mut E,
        forward: bool,
    ) {
        assert_eq!(self.inner_size, fft.size());
        twiddles.extend((0..self.inner_size).map(|i| {
            if let Some(index) = {
                if i < self.size {
                    Some((i as f64).powi(2))
                } else if i > self.inner_size - self.size {
                    Some(((i as f64) - (self.inner_size as f64)).powi(2))
                } else {
                    None
                }
            } {
                let twiddle = compute_half_twiddle(index, self.size);
                if forward {
                    twiddle
                } else {
                    twiddle.conj()
                }
            } else {
                Complex::default()
            }
        }));
        fft.fft_in_place(twiddles.as_mut());
    }

    pub fn initialize_x_twiddles<T: FftFloat, E: Extend<Complex<T>>>(
        &self,
        twiddles: &mut E,
        forward: bool,
    ) {
        twiddles.extend((0..self.size).map(|i| {
            let twiddle = compute_half_twiddle(-(i as f64).powi(2), self.size);
            if forward {
                twiddle
            } else {
                twiddle.conj()
            }
        }));
    }

    pub fn apply<T: FftFloat, F: Fft<Real = T>>(
        &self,
        input: &mut [Complex<T>],
        work: &mut [Complex<T>],
        x: &[Complex<T>],
        w: &[Complex<T>],
        fft: &F,
        transform: Transform,
    ) {
        assert_eq!(fft.size(), self.inner_size);
        assert_eq!(input.len(), self.size);
        assert_eq!(work.len(), self.inner_size);
        assert_eq!(x.len(), self.size);
        assert_eq!(w.len(), self.inner_size);
        apply(input, work, x, w, fft, transform);
    }
}

#[multiversion::target_clones("[x86|x86_64]+avx")]
#[inline]
fn apply<T: FftFloat, F: Fft<Real = T>>(
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
    x: &[Complex<T>],
    w: &[Complex<T>],
    fft: &F,
    transform: Transform,
) {
    let size = input.len();
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
    match transform {
        Transform::Fft | Transform::UnscaledIfft => {
            for (i, (w, xi)) in input.iter_mut().zip(work.iter().zip(x.iter())) {
                *i = w * xi;
            }
        }
        Transform::Ifft => {
            let scale = T::one() / T::from_usize(size).unwrap();
            for (i, (w, xi)) in input.iter_mut().zip(work.iter().zip(x.iter())) {
                *i = w * xi * scale;
            }
        }
        Transform::SqrtScaledFft | Transform::SqrtScaledIfft => {
            let scale = T::one() / T::sqrt(T::from_usize(size).unwrap());
            for (i, (w, xi)) in input.iter_mut().zip(work.iter().zip(x.iter())) {
                *i = w * xi * scale;
            }
        }
    }

    // TODO: this shouldn't be necessary...
    input[1..].reverse();
}
