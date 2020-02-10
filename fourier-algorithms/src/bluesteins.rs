use crate::{Autosort, Fft, FftFloat, Transform};
use core::cell::Cell;
use core::marker::PhantomData;
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

/// Initialize the "w" twiddles.
fn initialize_w_twiddles<
    T: FftFloat,
    E: Extend<Complex<T>> + AsMut<[Complex<T>]>,
    F: Fft<Real = T>,
>(
    size: usize,
    fft: &F,
    forward_twiddles: &mut E,
    inverse_twiddles: &mut E,
) {
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
            forward_twiddles.extend(core::iter::once(twiddle));
            inverse_twiddles.extend(core::iter::once(twiddle.conj()));
        } else {
            forward_twiddles.extend(core::iter::once(Complex::default()));
            inverse_twiddles.extend(core::iter::once(Complex::default()));
        }
    }
    fft.fft_in_place(forward_twiddles.as_mut());
    fft.fft_in_place(inverse_twiddles.as_mut());
}

/// Initialize the "x" twiddles.
fn initialize_x_twiddles<T: FftFloat, E: Extend<Complex<T>>>(
    size: usize,
    forward_twiddles: &mut E,
    inverse_twiddles: &mut E,
) {
    for i in 0..size {
        let twiddle = compute_half_twiddle(-(i as f64).powi(2), size);
        forward_twiddles.extend(core::iter::once(twiddle));
        inverse_twiddles.extend(core::iter::once(twiddle.conj()));
    }
}

/// Implements Bluestein's algorithm for arbitrary FFT sizes.
pub struct Bluesteins<T, InnerFft, Storage: Default> {
    size: usize,
    inner_fft: InnerFft,
    w_forward: Storage,
    w_inverse: Storage,
    x_forward: Storage,
    x_inverse: Storage,
    work: Cell<Storage>,
    real_type: PhantomData<T>,
}

impl<
        T: FftFloat,
        InnerFft: Fft<Real = T>,
        Storage: Default + Extend<Complex<T>> + AsMut<[Complex<T>]>,
    > Bluesteins<T, InnerFft, Storage>
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new_with_fft<F: Fn(usize) -> InnerFft>(size: usize, inner_fft_maker: F) -> Self {
        let inner_size = (2 * size - 1).checked_next_power_of_two().unwrap();
        let inner_fft = inner_fft_maker(inner_size);
        let mut w_forward = Storage::default();
        let mut w_inverse = Storage::default();
        let mut x_forward = Storage::default();
        let mut x_inverse = Storage::default();
        initialize_w_twiddles(size, &inner_fft, &mut w_forward, &mut w_inverse);
        initialize_x_twiddles(size, &mut x_forward, &mut x_inverse);
        let mut work = Storage::default();
        work.extend(core::iter::repeat(Complex::default()).take(inner_fft.size()));
        Self {
            size,
            inner_fft,
            w_forward,
            w_inverse,
            x_forward,
            x_inverse,
            work: Cell::new(work),
            real_type: PhantomData,
        }
    }
}

impl<Storage: Default + Extend<Complex<f32>> + AsRef<[Complex<f32>]> + AsMut<[Complex<f32>]>>
    Bluesteins<f32, Autosort<f32, Storage>, Storage>
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new(size: usize) -> Self {
        Self::new_with_fft(size, |size| Autosort::new(size).unwrap())
    }
}

impl<Storage: Default + Extend<Complex<f64>> + AsRef<[Complex<f64>]> + AsMut<[Complex<f64>]>>
    Bluesteins<f64, Autosort<f64, Storage>, Storage>
{
    /// Create a new Bluestein's algorithm generator.
    pub fn new(size: usize) -> Self {
        Self::new_with_fft(size, |size| Autosort::new(size).unwrap())
    }
}

impl<
        InnerFft: Fft<Real = f32>,
        Storage: Default + AsRef<[Complex<f32>]> + AsMut<[Complex<f32>]>,
    > Fft for Bluesteins<f32, InnerFft, Storage>
{
    type Real = f32;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<f32>], transform: Transform) {
        let mut work = self.work.take();
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
        self.work.set(work);
    }
}

impl<
        InnerFft: Fft<Real = f64>,
        Storage: Default + AsRef<[Complex<f64>]> + AsMut<[Complex<f64>]>,
    > Fft for Bluesteins<f64, InnerFft, Storage>
{
    type Real = f64;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<f64>], transform: Transform) {
        let mut work = self.work.take();
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
        self.work.set(work);
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

    // TODO: this shouldn't be necessary...
    input[1..].reverse();
}
