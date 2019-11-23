use super::radix2::Radix2;
use super::radix4::Radix4;
use super::Butterfly;
use crate::float::FftFloat;
use crate::vector::ComplexVector;
use num_complex::Complex;

pub struct Radix8<T> {
    twiddle: Complex<T>,
    radix2: Radix2,
    radix4: Radix4,
    forward: bool,
}

impl<T: FftFloat> Butterfly<T, 8> for Radix8<T> {
    fn new(forward: bool) -> Self {
        Self {
            twiddle: crate::twiddle::compute_twiddle(1, 8, forward),
            radix2: Radix2,
            radix4: Radix4::create(forward),
            forward,
        }
    }

    #[inline(always)]
    unsafe fn apply<Vector: ComplexVector<Float = T>>(&self, x: [Vector; 8]) -> [Vector; 8] {
        let twiddle = Vector::broadcast(&self.twiddle);
        let twiddle_neg = Vector::broadcast(&Complex::new(-self.twiddle.re, self.twiddle.im));
        let a1 = self.radix4.apply([x[0], x[2], x[4], x[6]]);
        let mut b1 = self.radix4.apply([x[1], x[3], x[5], x[7]]);
        b1[1] = b1[1].mul(&twiddle);
        b1[2] = b1[2].rotate(!self.forward);
        b1[3] = b1[3].mul(&twiddle_neg);
        let a2 = self.radix2.apply([a1[0], b1[0]]);
        let b2 = self.radix2.apply([a1[1], b1[1]]);
        let c2 = self.radix2.apply([a1[2], b1[2]]);
        let d2 = self.radix2.apply([a1[3], b1[3]]);
        [a2[0], b2[0], c2[0], d2[0], a2[1], b2[1], c2[1], d2[1]]
    }
}
