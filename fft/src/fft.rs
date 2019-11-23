use num_complex::Complex;

pub trait Fft {
    type Float;

    fn fft_in_place(&mut self, input: &mut [Complex<Self::Float>]);
    fn ifft_in_place(&mut self, input: &mut [Complex<Self::Float>]);
}
