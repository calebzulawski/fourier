//! Implementation of Bluestein's FFT algorithm.
use crate::{array::Array, autosort::Autosort, Fft, Float, Transform};
use core::cell::RefCell;
use core::marker::PhantomData;
use num_complex::Complex;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt, powi without std

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{boxed::Box, vec::Vec};

fn compute_half_twiddle<T: Float>(index: f64, size: usize) -> Complex<T> {
    let theta = index * core::f64::consts::PI / size as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}

/// Initialize the "w" twiddles.
fn initialize_w_twiddles<T: Float, F: Fft<Real = T>>(
    size: usize,
    fft: &F,
    forward_twiddles: &mut [Complex<T>],
    inverse_twiddles: &mut [Complex<T>],
) {
    assert_eq!(forward_twiddles.len(), fft.size());
    assert_eq!(inverse_twiddles.len(), fft.size());
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
            forward_twiddles[i] = twiddle.conj();
            inverse_twiddles[i] = twiddle;
        } else {
            forward_twiddles[i] = Complex::default();
            inverse_twiddles[i] = Complex::default();
        }
    }
    fft.fft_in_place(forward_twiddles);
    fft.fft_in_place(inverse_twiddles);
}

/// Initialize the "x" twiddles.
fn initialize_x_twiddles<T: Float>(
    size: usize,
    forward_twiddles: &mut [Complex<T>],
    inverse_twiddles: &mut [Complex<T>],
) {
    assert_eq!(forward_twiddles.len(), size);
    assert_eq!(inverse_twiddles.len(), size);
    for i in 0..size {
        let twiddle = compute_half_twiddle(-(i as f64).powi(2), size);
        forward_twiddles[i] = twiddle.conj();
        inverse_twiddles[i] = twiddle;
    }
}

/// Implements Bluestein's algorithm for arbitrary FFT sizes.
pub struct Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work> {
    size: usize,
    inner_fft: InnerFft,
    w_forward: WTwiddles,
    w_inverse: WTwiddles,
    x_forward: XTwiddles,
    x_inverse: XTwiddles,
    work: RefCell<Work>,
    real_type: PhantomData<T>,
}

impl<T, InnerFft, WTwiddles, XTwiddles, Work> core::fmt::Debug
    for Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work>
where
    InnerFft: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("Bluesteins")
            .field("size", &self.size)
            .field("inner_fft", &self.inner_fft)
            .finish()
    }
}

/// Returns the size of the inner FFT required for Bluestein's FFT.
pub fn inner_fft_size(size: usize) -> usize {
    (2 * size - 1).checked_next_power_of_two().unwrap()
}

impl<T, InnerFft, WTwiddles, XTwiddles, Work> Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work>
where
    T: Float,
    InnerFft: Fft<Real = T>,
    WTwiddles: Array<Complex<T>>,
    XTwiddles: Array<Complex<T>>,
    Work: Array<Complex<T>>,
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new_with_fft(
        size: usize,
        inner_fft: InnerFft,
        mut w_forward: WTwiddles,
        mut w_inverse: WTwiddles,
        mut x_forward: XTwiddles,
        mut x_inverse: XTwiddles,
        work: Work,
    ) -> Self {
        assert_eq!(inner_fft.size(), inner_fft_size(size));
        initialize_w_twiddles(size, &inner_fft, w_forward.as_mut(), w_inverse.as_mut());
        initialize_x_twiddles(size, x_forward.as_mut(), x_inverse.as_mut());
        Self {
            size,
            inner_fft,
            w_forward,
            w_inverse,
            x_forward,
            x_inverse,
            work: RefCell::new(work),
            real_type: PhantomData,
        }
    }
}

impl<T, InnerFft, WTwiddles, XTwiddles, Work> Fft
    for Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work>
where
    T: Float,
    InnerFft: Fft<Real = T>,
    WTwiddles: AsRef<[Complex<T>]>,
    XTwiddles: AsRef<[Complex<T>]>,
    Work: AsMut<[Complex<T>]>,
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

impl<T, WTwiddles, XTwiddles, Work, AutosortTwiddles, AutosortWork>
    Bluesteins<T, Autosort<T, AutosortTwiddles, AutosortWork>, WTwiddles, XTwiddles, Work>
where
    T: Float,
    WTwiddles: Array<Complex<T>>,
    XTwiddles: Array<Complex<T>>,
    Work: Array<Complex<T>>,
    AutosortTwiddles: Array<Complex<T>>,
    AutosortWork: Array<Complex<T>>,
    Autosort<T, AutosortTwiddles, AutosortWork>: Fft<Real = T>,
{
    /// Constructs an FFT over types that are `Extend`.
    pub fn new(size: usize) -> Self {
        let inner_fft_size = inner_fft_size(size);
        let inner_fft = Autosort::new(inner_fft_size);
        Self::new_with_fft(
            size,
            inner_fft.unwrap(),
            Array::new(inner_fft_size),
            Array::new(inner_fft_size),
            Array::new(size),
            Array::new(size),
            Array::new(inner_fft_size),
        )
    }
}

/// Implementation of Bluestein's algorithm backed by heap allocations.
///
/// Requires the `std` or `alloc` features.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type HeapBluesteins<T> = Bluesteins<
    T,
    Autosort<T, Box<[Complex<T>]>, Box<[Complex<T>]>>,
    Box<[Complex<T>]>,
    Box<[Complex<T>]>,
    Box<[Complex<T>]>,
>;

#[multiversion::multiversion]
#[clone(target = "[x86|x86_64]+avx")]
#[inline]
fn apply<T: Float, F: Fft<Real = T>>(
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
