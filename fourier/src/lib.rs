//! This crate provides fast Fourier transforms (FFT) in pure Rust.
//!
//! # Implementation
//! For FFTs with sizes that are multiples of 2 and 3, the Stockham auto-sort algorithm is used.
//! For any other sizes, Bluestein's algorithm is used.

#[macro_use]
mod vector;

mod autosort;
mod bluesteins;
mod fft;
mod float;
mod twiddle;

pub use crate::fft::*;

/// Create a complex-valued FFT over `f32` with the specified size.
pub fn create_fft_f32(size: usize) -> impl Fft<Real = f32> + Send {
    if let Some(fft) = crate::autosort::prime_factor::create_f32(size) {
        Either::Left(fft)
    } else {
        Either::Right(crate::bluesteins::create_f32(size))
    }
}

/// Create a complex-valued FFT over `f64` with the specified size.
pub fn create_fft_f64(size: usize) -> impl Fft<Real = f64> + Send {
    if let Some(fft) = crate::autosort::prime_factor::create_f64(size) {
        Either::Left(fft)
    } else {
        Either::Right(crate::bluesteins::create_f64(size))
    }
}
