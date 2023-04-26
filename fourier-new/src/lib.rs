#![feature(portable_simd)]

mod autosort;
mod scalar;

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
