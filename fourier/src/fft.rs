use num_complex::Complex;

/// Trait for applying an FFT
pub trait Fft {
    /// The real type used by the FFT.
    type Real: Copy;

    /// The size of the FFT.
    fn size(&self) -> usize;

    /// Apply an FFT or IFFT in-place.
    fn transform_in_place(&self, input: &mut [Complex<Self::Real>], forward: bool);

    /// Apply an FFT or IFFT out-of-place.
    fn transform(
        &self,
        input: &[Complex<Self::Real>],
        output: &mut [Complex<Self::Real>],
        forward: bool,
    ) {
        assert_eq!(input.len(), self.size());
        assert_eq!(output.len(), self.size());
        output.copy_from_slice(input);
        self.transform_in_place(output, forward);
    }

    /// Apply an FFT in-place.
    fn fft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, true);
    }

    /// Apply an IFFT in-place.
    fn ifft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, false);
    }

    /// Apply an FFT out-of-place.
    fn fft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, true);
    }

    /// Apply an IFFT out-of-place.
    fn ifft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, false);
    }
}
