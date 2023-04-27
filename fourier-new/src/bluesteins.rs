use crate::{autosort::Autosort, scalar::Scalar, Fft, Transform};
use core::cell::RefCell;
use num_complex::Complex;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt, powi without std

fn compute_half_twiddle<T: Scalar>(index: f64, size: usize) -> Complex<T> {
    let theta = index * core::f64::consts::PI / size as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}

/// Initialize the "w" twiddles.
fn initialize_w_twiddles<T: Scalar, F: Fft<Real = T>>(
    size: usize,
    fft: &F,
) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
    let mut forward_twiddles = Vec::new();
    let mut inverse_twiddles = Vec::new();
    for i in 0..fft.size() {
        if let Some(index) = {
            if i < size {
                Some((i as f64).powi(2))
            } else if i > fft.size() - size {
                Some(((i as f64) - (fft.size() as f64)).powi(2))
            } else {
                None
            }
        } {
            let twiddle = compute_half_twiddle(index, size);
            forward_twiddles.push(twiddle.conj());
            inverse_twiddles.push(twiddle);
        } else {
            forward_twiddles.push(Complex::default());
            inverse_twiddles.push(Complex::default());
        }
    }
    fft.fft_in_place(forward_twiddles.as_mut());
    fft.fft_in_place(inverse_twiddles.as_mut());
    (forward_twiddles, inverse_twiddles)
}

/// Initialize the "x" twiddles.
fn initialize_x_twiddles<T: Scalar>(size: usize) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
    let mut forward_twiddles = Vec::new();
    let mut inverse_twiddles = Vec::new();
    for i in 0..size {
        let twiddle = compute_half_twiddle(-(i as f64).powi(2), size);
        forward_twiddles.push(twiddle.conj());
        inverse_twiddles.push(twiddle);
    }
    (forward_twiddles, inverse_twiddles)
}

/// Implements Bluestein's algorithm for arbitrary FFT sizes.
pub struct Bluesteins<T> {
    size: usize,
    inner_fft: Autosort<T>,
    w_forward: Vec<Complex<T>>,
    w_inverse: Vec<Complex<T>>,
    x_forward: Vec<Complex<T>>,
    x_inverse: Vec<Complex<T>>,
    work: RefCell<Vec<Complex<T>>>,
}

impl<T: Scalar> Bluesteins<T>
where
    Autosort<T>: Fft<Real = T>,
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new(size: usize) -> Self {
        let inner_size = (2 * size - 1).checked_next_power_of_two().unwrap();
        let inner_fft = Autosort::new(inner_size).unwrap();
        let (w_forward, w_inverse) = initialize_w_twiddles(size, &inner_fft);
        let (x_forward, x_inverse) = initialize_x_twiddles(size);
        let work = vec![Complex::default(); inner_size];
        Self {
            size,
            inner_fft,
            w_forward,
            w_inverse,
            x_forward,
            x_inverse,
            work: RefCell::new(work),
        }
    }
}

impl<T> Fft for Bluesteins<T>
where
    T: Scalar,
    Autosort<T>: Fft<Real = T>,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<T>], transform: Transform) {
        let mut work = self.work.borrow_mut();
        let (x, w) = if transform.is_forward() {
            (&self.x_forward, &self.w_forward)
        } else {
            (&self.x_inverse, &self.w_inverse)
        };
        apply(
            input,
            work.as_mut(),
            x.as_ref(),
            w.as_ref(),
            &self.inner_fft,
            transform,
        );
    }
}

fn apply<T: Scalar, F: Fft<Real = T>>(
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
    x: &[Complex<T>],
    w: &[Complex<T>],
    fft: &F,
    transform: Transform,
) {
    assert_eq!(x.len(), input.len());

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
}
