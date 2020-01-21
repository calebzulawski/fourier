use crate::float::FftFloat;
use crate::{Fft, Transform};
use num_complex::Complex;
use std::cell::Cell;

fn compute_half_twiddle<T: FftFloat>(index: f64, size: usize) -> Complex<T> {
    let theta = index * std::f64::consts::PI / size as f64;
    Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    )
}

struct BluesteinsAlgorithm<F, T>
where
    F: Fft<Real = T> + Send,
    T: FftFloat,
{
    fft: F,
    size: usize,
    w_forward: Box<[Complex<T>]>,
    w_inverse: Box<[Complex<T>]>,
    x_forward: Box<[Complex<T>]>,
    x_inverse: Box<[Complex<T>]>,
    work: Cell<Box<[Complex<T>]>>,
}

impl<F, T> BluesteinsAlgorithm<F, T>
where
    F: Fft<Real = T> + Send,
    T: FftFloat,
{
    fn new<M>(size: usize, fft_maker: M) -> Self
    where
        M: Fn(usize) -> F,
    {
        let fft = fft_maker((2 * size - 1).checked_next_power_of_two().unwrap());

        // create W vector
        let mut w_forward = vec![Complex::default(); fft.size()].into_boxed_slice();
        let mut w_inverse = vec![Complex::default(); fft.size()].into_boxed_slice();
        for (i, (wfi, wii)) in w_forward.iter_mut().zip(w_inverse.iter_mut()).enumerate() {
            if let Some(index) = {
                if i < size {
                    Some((i as f64).powi(2))
                } else if i > fft.size() - size {
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
            work: Cell::new(vec![Complex::default(); fft.size()].into_boxed_slice()),
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
    fft: &(dyn Fft<Real = T> + Send),
    transform: Transform,
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

impl<F, T> Fft for BluesteinsAlgorithm<F, T>
where
    F: Fft<Real = T> + Send,
    T: FftFloat,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<Self::Real>], transform: Transform) {
        let mut work = self.work.take();
        apply(
            input,
            &mut work,
            if transform.is_forward() {
                &self.x_forward
            } else {
                &self.x_inverse
            },
            if transform.is_forward() {
                &self.w_forward
            } else {
                &self.w_inverse
            },
            self.size,
            &self.fft,
            transform,
        );
        self.work.set(work);
    }
}

pub fn create_f32(size: usize) -> impl Fft<Real = f32> + Send {
    BluesteinsAlgorithm::new(size, |size| {
        crate::autosort::prime_factor::create_f32(size).unwrap()
    })
}

pub fn create_f64(size: usize) -> impl Fft<Real = f64> + Send {
    BluesteinsAlgorithm::new(size, |size| {
        crate::autosort::prime_factor::create_f64(size).unwrap()
    })
}
