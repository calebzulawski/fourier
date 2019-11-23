use super::radix2::Radix2;
use super::Butterfly;
use crate::float::FftFloat;
use crate::vector::ComplexVector;

pub struct Radix4 {
    radix2: Radix2,
    forward: bool,
}

impl Radix4 {
    pub fn create(forward: bool) -> Self {
        Self {
            radix2: Radix2,
            forward,
        }
    }
}

impl<T: FftFloat> Butterfly<T, 4> for Radix4 {
    fn new(forward: bool) -> Self {
        Radix4::create(forward)
    }

    #[inline(always)]
    unsafe fn apply<Vector: ComplexVector<Float = T>>(&self, x: [Vector; 4]) -> [Vector; 4] {
        let a1 = self.radix2.apply([x[0], x[2]]);
        let mut b1 = self.radix2.apply([x[1], x[3]]);
        b1[1] = b1[1].rotate(self.forward);
        let a2 = self.radix2.apply([a1[0], b1[0]]);
        let b2 = self.radix2.apply([a1[1], b1[1]]);
        [a2[0], b2[1], a2[1], b2[0]]
    }
}
