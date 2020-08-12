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

pub use fourier_algorithms::{stack_fft, Fft, Transform};

/// A real scalar type that supports FFTs.
///
/// Requires the `std` or `alloc` feature.
#[cfg(any(feature = "std", feature = "alloc"))]
pub trait Float: Copy {
    fn create_fft(size: usize) -> Box<dyn Fft<Real = Self> + Send>;
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl Float for f32 {
    fn create_fft(size: usize) -> Box<dyn Fft<Real = Self> + Send> {
        Box::new(fourier_algorithms::HeapAlgorithm::new(size))
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl Float for f64 {
    fn create_fft(size: usize) -> Box<dyn Fft<Real = Self> + Send> {
        Box::new(fourier_algorithms::HeapAlgorithm::new(size))
    }
}

/// Create a complex-valued FFT over `T` with the specified size.
///
/// Requires the `std` or `alloc` feature.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn create_fft<T>(size: usize) -> Box<dyn Fft<Real = T> + Send>
where
    T: Float,
{
    T::create_fft(size)
}
