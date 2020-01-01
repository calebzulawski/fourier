#[macro_use]
mod vector;

mod autosort;
mod bluesteins;
mod fft;
mod float;
mod twiddle;

use crate::autosort::prime_factor::PrimeFactorFft32;
pub use crate::fft::Fft;

pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Float = f32>> {
    if size.is_power_of_two() {
        Box::new(PrimeFactorFft32::new(size))
    } else {
        crate::bluesteins::create_f32(size)
    }
}
