//! Implementation of Bluestein's FFT algorithm.
use crate::{autosort::Autosort, Fft, Float, Transform};
use core::cell::RefCell;
use core::marker::PhantomData;
use num_complex::Complex;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt, powi without std

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

/*
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

impl<T, InnerFft, WTwiddles, XTwiddles, Work> Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work> {
    /// Create a new transform generator from parts.  Twiddles factors and work must be the correct
    /// size.
    pub unsafe fn new_from_parts(
        size: usize,
        inner_fft: InnerFft,
        w_forward: WTwiddles,
        w_inverse: WTwiddles,
        x_forward: XTwiddles,
        x_inverse: XTwiddles,
        work: Work,
    ) -> Self {
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

impl<
        T: Float,
        InnerFft: Fft<Real = T>,
        WTwiddles: Default + Extend<Complex<T>> + AsMut<[Complex<T>]>,
        XTwiddles: Default + Extend<Complex<T>>,
        Work: Default + Extend<Complex<T>>,
    > Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work>
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new_with_fft<F: Fn(usize) -> InnerFft>(size: usize, inner_fft_maker: F) -> Self {
        let inner_size = (2 * size - 1).checked_next_power_of_two().unwrap();
        let inner_fft = inner_fft_maker(inner_size);
        let mut w_forward = WTwiddles::default();
        let mut w_inverse = WTwiddles::default();
        let mut x_forward = XTwiddles::default();
        let mut x_inverse = XTwiddles::default();
        initialize_w_twiddles(size, &inner_fft, &mut w_forward, &mut w_inverse);
        initialize_x_twiddles(size, &mut x_forward, &mut x_inverse);
        let mut work = Work::default();
        work.extend(core::iter::repeat(Complex::default()).take(inner_fft.size()));
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

impl<
        T: Float,
        InnerFft: Fft<Real = T>,
        WTwiddles: AsRef<[Complex<T>]>,
        XTwiddles: AsRef<[Complex<T>]>,
        Work: AsRef<[Complex<T>]>,
    > Bluesteins<T, InnerFft, WTwiddles, XTwiddles, Work>
{
    /// Return the w-twiddle factors.
    pub fn w_twiddles(&self) -> (&[Complex<T>], &[Complex<T>]) {
        (self.w_forward.as_ref(), self.w_inverse.as_ref())
    }

    /// Return the w-twiddle factors.
    pub fn x_twiddles(&self) -> (&[Complex<T>], &[Complex<T>]) {
        (self.x_forward.as_ref(), self.x_inverse.as_ref())
    }

    /// Return the inner FFT size.
    pub fn inner_fft_size(&self) -> usize {
        self.inner_fft.size()
    }

    /// Return the work buffer size.
    pub fn work_size(&self) -> usize {
        self.work.borrow().as_ref().len()
    }
}

macro_rules! implement {
    {
        $type:ty
    } => {
        impl<
                AutosortTwiddles: Default + Extend<Complex<$type>> + AsRef<[Complex<$type>]>,
                AutosortWork: Default + Extend<Complex<$type>> + AsMut<[Complex<$type>]>,
                WTwiddles: Default + Extend<Complex<$type>> + AsMut<[Complex<$type>]>,
                XTwiddles: Default + Extend<Complex<$type>>,
                Work: Default + Extend<Complex<$type>>,
            > Bluesteins<$type, Autosort<$type, AutosortTwiddles, AutosortWork>, WTwiddles, XTwiddles, Work>
        {
            /// Create a new Bluestein's algorithm generator.
            pub fn new(size: usize) -> Self {
                Self::new_with_fft(size, |size| Autosort::new(size).unwrap())
            }
        }

        impl<
                InnerFft: Fft<Real = $type>,
                WTwiddles: AsRef<[Complex<$type>]>,
                XTwiddles: AsRef<[Complex<$type>]>,
                Work: AsMut<[Complex<$type>]>,
            > Fft for Bluesteins<$type, InnerFft, WTwiddles, XTwiddles, Work>
        {
            type Real = $type;

            fn size(&self) -> usize {
                self.size
            }

            fn transform_in_place(&self, input: &mut [Complex<$type>], transform: Transform) {
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
    }
}
implement! { f32 }
implement! { f64 }

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
*/
