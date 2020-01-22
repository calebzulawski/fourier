use crate::float;
use num_complex::Complex;

/// Specifies a type of transform to perform.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Transform {
    /// Fast Fourier transform
    Fft,
    /// Inverse fast Fourier transform
    Ifft,
    /// Unscaled IFFT (conventionally the IFFT is scaled by `1 / N`)
    UnscaledIfft,
    /// Square-root scaled FFT (a unitary transform)
    SqrtScaledFft,
    /// Square-root scaled IFFT (a unitary transform)
    SqrtScaledIfft,
}

impl Transform {
    /// Returns true if the transform is a forward transform.
    pub fn is_forward(&self) -> bool {
        match self {
            Self::Fft | Self::SqrtScaledFft => true,
            Self::Ifft | Self::UnscaledIfft | Self::SqrtScaledIfft => false,
        }
    }

    /// Returns the inverse transform, or `None` for `UnscaledIfft`.
    pub fn inverse(&self) -> Option<Self> {
        match self {
            Self::Fft => Some(Self::Ifft),
            Self::Ifft => Some(Self::Fft),
            Self::SqrtScaledFft => Some(Self::SqrtScaledIfft),
            Self::SqrtScaledIfft => Some(Self::SqrtScaledFft),
            Self::UnscaledIfft => None,
        }
    }
}

/// The interface for performing FFTs.
pub trait Fft {
    /// The real type used by the FFT.
    type Real: Copy;

    /// The size of the FFT.
    fn size(&self) -> usize;

    /// Apply an FFT or IFFT in-place.
    fn transform_in_place(&self, input: &mut [Complex<Self::Real>], transform: Transform);

    /// Apply an FFT or IFFT out-of-place.
    fn transform(
        &self,
        input: &[Complex<Self::Real>],
        output: &mut [Complex<Self::Real>],
        transform: Transform,
    ) {
        assert_eq!(input.len(), self.size());
        assert_eq!(output.len(), self.size());
        output.copy_from_slice(input);
        self.transform_in_place(output, transform);
    }

    /// Apply an FFT in-place.
    fn fft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, Transform::Fft);
    }

    /// Apply an IFFT in-place.
    fn ifft_in_place(&self, input: &mut [Complex<Self::Real>]) {
        self.transform_in_place(input, Transform::Ifft);
    }

    /// Apply an FFT out-of-place.
    fn fft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, Transform::Fft);
    }

    /// Apply an IFFT out-of-place.
    fn ifft(&self, input: &[Complex<Self::Real>], output: &mut [Complex<Self::Real>]) {
        self.transform(input, output, Transform::Ifft);
    }
}

pub(crate) enum Fallback<F1, F2> {
    Primary(F1),
    Secondary(F2),
}

impl<F1, F2, R> Fft for Fallback<F1, F2>
where
    F1: Fft<Real = R>,
    F2: Fft<Real = R>,
    R: float::FftFloat,
{
    type Real = R;

    fn size(&self) -> usize {
        match *self {
            Fallback::Primary(ref f1) => f1.size(),
            Fallback::Secondary(ref f2) => f2.size(),
        }
    }

    fn transform_in_place(&self, input: &mut [Complex<Self::Real>], transform: Transform) {
        match *self {
            Fallback::Primary(ref f1) => f1.transform_in_place(input, transform),
            Fallback::Secondary(ref f2) => f2.transform_in_place(input, transform),
        }
    }
}
