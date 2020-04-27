//! Size 1 FFTs.

use crate::{Fft, Transform};
use num_complex::Complex;

/// Implements a size 1 FFT.
#[derive(Default, Copy, Clone)]
pub struct Identity<T> {
    phantom_data: core::marker::PhantomData<T>,
}

impl<T> core::fmt::Debug for Identity<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        write!(f, "Identity")
    }
}

impl<T> Fft for Identity<T>
where
    T: Copy,
{
    type Real = T;

    fn size(&self) -> usize {
        1
    }

    fn transform_in_place(&self, _input: &mut [Complex<Self::Real>], _transform: Transform) {}
}
