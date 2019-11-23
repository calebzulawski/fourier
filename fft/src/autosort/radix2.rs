use super::Butterfly;
use crate::float::FftFloat;
use crate::vector::ComplexVector;

pub struct Radix2;

impl<T: FftFloat> Butterfly<T, 2> for Radix2 {
    fn new(_forward: bool) -> Self {
        Self
    }

    #[inline(always)]
    unsafe fn apply<Vector: ComplexVector<Float = T>>(&self, x: [Vector; 2]) -> [Vector; 2] {
        [x[0].add(&x[1]), x[0].sub(&x[1])]
    }
}
