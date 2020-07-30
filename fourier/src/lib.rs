//! This crate provides fast Fourier transforms (FFT) in pure Rust.
//!
//! # Implementation
//! For FFTs with sizes that are multiples of 2 and 3, the Stockham auto-sort algorithm is used.
//! For any other sizes, Bluestein's algorithm is used.
//!
//! # Optional features
//! Fourier uses optional features to allow versatility with `#[no_std]`.
//! -  **`std`** *(enabled by default)* - Uses heap allocation for runtime-sized FFTs.  Enables
//!    runtime CPU feature detection and dispatch.  If disabled, only compile-time CPU feature
//!    detection is performed.
//! -  **`alloc`** - Enables heap allocation for runtime-sized FFTs with `#[no_std]` using the
//!    [`alloc`] crate.
//!
//! [`alloc`]: https://doc.rust-lang.org/alloc/
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::boxed::Box;

pub use fourier_algorithms::{Fft, Transform};
//pub use fourier_macros::static_fft;

/// Create a complex-valued FFT over `f32` with the specified size.
///
/// Requires the `std` or `alloc` feature.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Real = f32> + Send> {
    use fourier_algorithms::{
        autosort::HeapAutosort, bluesteins::HeapBluesteins, identity::Identity,
    };
    if size == 1 {
        Box::new(Identity::default())
    } else if let Some(fft) = HeapAutosort::<f32>::new(size) {
        Box::new(fft)
    } else {
        Box::new(HeapBluesteins::new(size))
    }
}

/// Create a complex-valued FFT over `f64` with the specified size.
///
/// Requires the `std` or `alloc` feature.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft_f64(size: usize) -> Box<dyn Fft<Real = f64> + Send> {
    use fourier_algorithms::{
        autosort::HeapAutosort, bluesteins::HeapBluesteins, identity::Identity,
    };
    if size == 1 {
        Box::new(Identity::default())
    } else if let Some(fft) = HeapAutosort::<f64>::new(size) {
        Box::new(fft)
    } else {
        Box::new(HeapBluesteins::new(size))
    }
}
