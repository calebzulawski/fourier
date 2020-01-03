#[macro_use]
mod vector;

mod autosort;
mod bluesteins;
mod fft;
mod float;
mod twiddle;

pub use crate::fft::Fft;

pub fn create_fft_f32(size: usize) -> Box<dyn Fft<Real = f32> + Send> {
    if let Some(fft) = crate::autosort::prime_factor::create_f32(size) {
        fft
    } else {
        crate::bluesteins::create_f32(size)
    }
}
