#[macro_use]
mod vector;

mod autosort;
mod fft;
mod float;
mod twiddle;

use crate::autosort::prime_factor::PrimeFactorFft32;
pub use crate::fft::Fft;

pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Float = f32>> {
    Box::new(PrimeFactorFft32::new(size))
    /*
    if let Some(fft) = PowerTwoFft32::new(size) {
        Box::new(fft)
    } else {
        Box::new(PrimeFactorFft32::new(size))
    }
    */
}
