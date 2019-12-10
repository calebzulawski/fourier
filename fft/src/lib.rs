#![feature(const_generics)]
#![feature(const_if_match)]

mod autosort;
mod fft;
mod float;
mod twiddle;
mod vector;

use crate::autosort::pow2::PowerTwoFft32;
use crate::autosort::prime_factor::PrimeFactorFft32;
pub use crate::fft::Fft;

pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Float = f32>> {
    if let Some(fft) = PowerTwoFft32::new(size) {
        Box::new(fft)
    } else {
        Box::new(PrimeFactorFft32::new(size))
    }
}
