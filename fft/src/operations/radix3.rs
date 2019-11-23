use super::Butterfly;
use crate::float::FftFloat;
use crate::vector::ComplexVector;
use num_complex::Complex;

pub struct Radix3<T> {
    twiddle: Complex<T>,
}

impl<T: FftFloat> Butterfly<T, 3> for Radix3<T> {
    fn new(forward: bool) -> Self {
        Self {
            twiddle: crate::twiddle::compute_twiddle(1, 3, forward),
        }
    }

    #[inline(always)]
    unsafe fn apply<Vector: ComplexVector<Float = T>>(&self, x: [Vector; 3]) -> [Vector; 3] {
        let twiddle = Vector::broadcast(&self.twiddle);
        let twiddle_conj = Vector::broadcast(&self.twiddle.conj());
        [
            x[0].add(&x[1].add(&x[2])),
            x[0].add(&x[1].mul(&twiddle).add(&x[2].mul(&twiddle_conj))),
            x[0].add(&x[1].mul(&twiddle_conj).add(&x[2].mul(&twiddle))),
        ]
    }
}
