use super::ComplexVector;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Generic<T>(Complex<T>);

impl<T: FftFloat + Default> ComplexVector for Generic<T> {
    type Float = T;
    const WIDTH: usize = 1;

    unsafe fn zero() -> Self {
        Self(Complex::default())
    }
    unsafe fn broadcast(value: &Complex<Self::Float>) -> Self {
        Self(*value)
    }

    unsafe fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
    unsafe fn sub(&self, rhs: &Self) -> Self {
        Self(self.0 - rhs.0)
    }
    unsafe fn mul(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }
    unsafe fn rotate(&self, positive: bool) -> Self {
        Self(if positive {
            Complex::new(-self.0.im, self.0.re)
        } else {
            Complex::new(self.0.im, -self.0.re)
        })
    }

    unsafe fn load(from: *const Complex<Self::Float>) -> Self {
        Self(*from)
    }
    unsafe fn store(&self, to: *mut Complex<Self::Float>) {
        *to = self.0;
    }

    unsafe fn partial_load(_from: *const Complex<Self::Float>, _count: usize) -> Self {
        unimplemented!("cannot partial load a generic vector")
    }
    unsafe fn partial_store(&self, _to: *mut Complex<Self::Float>, _count: usize) {
        unimplemented!("cannot partial store a generic vector")
    }
}
