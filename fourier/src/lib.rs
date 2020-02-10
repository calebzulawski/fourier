//! This crate provides fast Fourier transforms (FFT) in pure Rust.
//!
//! # Implementation
//! For FFTs with sizes that are multiples of 2 and 3, the Stockham auto-sort algorithm is used.
//! For any other sizes, Bluestein's algorithm is used.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Vec;

pub use fourier_algorithms::{Fft, Transform};
pub use fourier_macros::static_fft;

/// Create a complex-valued FFT over `f32` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Real = f32> + Send> {
    use fourier_algorithms::{Autosort, Bluesteins};
    use num_complex::Complex;
    type Autosort32 = Autosort<f32, Vec<Complex<f32>>, Vec<Complex<f32>>>;
    type Bluesteins32 =
        Bluesteins<f32, Autosort32, Vec<Complex<f32>>, Vec<Complex<f32>>, Vec<Complex<f32>>>;
    if let Some(fft) = Autosort32::new(size) {
        Box::new(fft)
    } else {
        Box::new(Bluesteins32::new(size))
    }
}

/// Create a complex-valued FFT over `f64` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f64(size: usize) -> Box<dyn Fft<Real = f64> + Send> {
    use fourier_algorithms::{Autosort, Bluesteins};
    use num_complex::Complex;
    type Autosort64 = Autosort<f64, Vec<Complex<f64>>, Vec<Complex<f64>>>;
    type Bluesteins64 =
        Bluesteins<f64, Autosort64, Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>>;
    if let Some(fft) = Autosort64::new(size) {
        Box::new(fft)
    } else {
        Box::new(Bluesteins64::new(size))
    }
}
