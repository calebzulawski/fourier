use num_complex::Complex;

pub trait Fft {
    type Real: Copy;

    fn size(&self) -> usize;

    fn transform_in_place(&self, input: &mut [Complex<Self::Real>], forward: bool);

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

    fn fft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, true);
    }
    fn ifft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, false);
    }
    fn fft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, true);
    }
    fn ifft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, false);
    }
}
