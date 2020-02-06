//! This crate provides fast Fourier transforms (FFT) in pure Rust.
//!
//! # Implementation
//! For FFTs with sizes that are multiples of 2 and 3, the Stockham auto-sort algorithm is used.
//! For any other sizes, Bluestein's algorithm is used.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

pub use fourier_algorithms::fft::{Fft, Transform};

#[cfg(any(feature = "std", feature = "alloc"))]
mod has_alloc {
    #[cfg(not(feature = "std"))]
    use alloc::boxed::{Box, Vec};

    use super::{Fft, Transform};
    use core::cell::Cell;
    use fourier_algorithms::{autosort::Stages, bluesteins::Bluesteins, float::FftFloat};
    use num_complex::Complex;

    pub(crate) struct AutosortFft<T> {
        stages: Stages,
        work: Cell<Box<[Complex<T>]>>,
        forward_twiddles: Box<[Complex<T>]>,
        inverse_twiddles: Box<[Complex<T>]>,
    }

    impl<T: FftFloat> AutosortFft<T> {
        pub fn new(size: usize) -> Option<Self> {
            if let Some(stages) = fourier_algorithms::autosort::Stages::new(size) {
                let mut forward_twiddles = Vec::new();
                let mut inverse_twiddles = Vec::new();
                stages.initialize_twiddles(&mut forward_twiddles, true);
                stages.initialize_twiddles(&mut inverse_twiddles, false);
                Some(Self {
                    stages,
                    work: Cell::new(vec![Complex::default(); size].into_boxed_slice()),
                    forward_twiddles: forward_twiddles.into_boxed_slice(),
                    inverse_twiddles: inverse_twiddles.into_boxed_slice(),
                })
            } else {
                None
            }
        }
    }

    impl Fft for AutosortFft<f32> {
        type Real = f32;

        fn size(&self) -> usize {
            self.stages.size()
        }

        fn transform_in_place(&self, input: &mut [Complex<f32>], transform: Transform) {
            let mut work = self.work.take();
            let twiddles = if transform.is_forward() {
                &self.forward_twiddles
            } else {
                &self.inverse_twiddles
            };
            self.stages.apply_f32(input, &mut work, twiddles, transform);
            self.work.set(work);
        }
    }

    impl Fft for AutosortFft<f64> {
        type Real = f64;

        fn size(&self) -> usize {
            self.stages.size()
        }

        fn transform_in_place(&self, input: &mut [Complex<f64>], transform: Transform) {
            let mut work = self.work.take();
            let twiddles = if transform.is_forward() {
                &self.forward_twiddles
            } else {
                &self.inverse_twiddles
            };
            self.stages.apply_f64(input, &mut work, twiddles, transform);
            self.work.set(work);
        }
    }

    pub(crate) struct BluesteinsFft<T: FftFloat, F: Fft<Real = T>> {
        forward_x: Box<[Complex<T>]>,
        inverse_x: Box<[Complex<T>]>,
        forward_w: Box<[Complex<T>]>,
        inverse_w: Box<[Complex<T>]>,
        work: Cell<Box<[Complex<T>]>>,
        fft: F,
        bluesteins: Bluesteins,
    }

    impl<T: FftFloat, F: Fft<Real = T>> BluesteinsFft<T, F> {
        fn new_impl<Func: FnOnce(usize) -> F>(size: usize, fft_maker: Func) -> Self {
            let bluesteins = Bluesteins::new(size);
            let fft = fft_maker(bluesteins.inner_fft_size());
            let mut forward_x = Vec::new();
            let mut inverse_x = Vec::new();
            bluesteins.initialize_x_twiddles(&mut forward_x, true);
            bluesteins.initialize_x_twiddles(&mut inverse_x, false);
            let mut forward_w = Vec::new();
            let mut inverse_w = Vec::new();
            bluesteins.initialize_w_twiddles(&fft, &mut forward_w, true);
            bluesteins.initialize_w_twiddles(&fft, &mut inverse_w, false);
            Self {
                forward_x: forward_x.into_boxed_slice(),
                inverse_x: inverse_x.into_boxed_slice(),
                forward_w: forward_w.into_boxed_slice(),
                inverse_w: inverse_w.into_boxed_slice(),
                work: Cell::new(
                    vec![Complex::default(); bluesteins.inner_fft_size()].into_boxed_slice(),
                ),
                fft,
                bluesteins,
            }
        }
    }

    impl BluesteinsFft<f32, AutosortFft<f32>> {
        pub fn new(size: usize) -> Self {
            Self::new_impl(size, |size| AutosortFft::new(size).unwrap())
        }
    }

    impl BluesteinsFft<f64, AutosortFft<f64>> {
        pub fn new(size: usize) -> Self {
            Self::new_impl(size, |size| AutosortFft::new(size).unwrap())
        }
    }

    impl<T: FftFloat, F: Fft<Real = T>> Fft for BluesteinsFft<T, F> {
        type Real = T;

        fn size(&self) -> usize {
            self.bluesteins.size()
        }

        fn transform_in_place(&self, input: &mut [Complex<T>], transform: Transform) {
            let mut work = self.work.take();
            let (x, w) = if transform.is_forward() {
                (&self.forward_x, &self.forward_w)
            } else {
                (&self.inverse_x, &self.inverse_w)
            };
            self.bluesteins
                .apply(input, &mut work, x, w, &self.fft, transform);
            self.work.set(work);
        }
    }
}

/// Create a complex-valued FFT over `f32` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Real = f32> + Send> {
    if let Some(fft) = has_alloc::AutosortFft::<f32>::new(size) {
        Box::new(fft)
    } else {
        Box::new(has_alloc::BluesteinsFft::<f32, has_alloc::AutosortFft<f32>>::new(size))
    }
}

/// Create a complex-valued FFT over `f64` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f64(size: usize) -> Box<dyn Fft<Real = f64> + Send> {
    if let Some(fft) = has_alloc::AutosortFft::<f64>::new(size) {
        Box::new(fft)
    } else {
        Box::new(has_alloc::BluesteinsFft::<f64, has_alloc::AutosortFft<f64>>::new(size))
    }
}
