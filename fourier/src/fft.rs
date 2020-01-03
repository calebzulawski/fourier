use num_complex::Complex;

pub trait Fft {
    type Float;

    fn size(&self) -> usize;
    fn fft_in_place(&self, input: &mut [Complex<Self::Float>]);
    fn ifft_in_place(&self, input: &mut [Complex<Self::Float>]);
}
