//! This crate provides fast Fourier transforms (FFT) in pure Rust.
//!
//! # Implementation
//! For FFTs with sizes that are multiples of 2 and 3, the Stockham auto-sort algorithm is used.
//! For any other sizes, Bluestein's algorithm is used.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[macro_use]
mod vector;

#[cfg(any(feature = "std", feature = "alloc"))]
mod autosort;
#[cfg(any(feature = "std", feature = "alloc"))]
mod bluesteins;
mod fft;
mod float;
#[cfg(any(feature = "std", feature = "alloc"))]
mod twiddle;

pub use crate::fft::*;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::boxed::Box;

/// Create a complex-valued FFT over `f32` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Real = f32> + Send> {
    if let Some(fft) = crate::autosort::prime_factor::create_f32(size) {
        fft
    } else {
        crate::bluesteins::create_f32(size)
    }
}

/// Create a complex-valued FFT over `f64` with the specified size.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f64(size: usize) -> Box<dyn Fft<Real = f64> + Send> {
    if let Some(fft) = crate::autosort::prime_factor::create_f64(size) {
        fft
    } else {
        crate::bluesteins::create_f64(size)
    }
}
