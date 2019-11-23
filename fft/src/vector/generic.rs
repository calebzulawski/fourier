use super::ComplexVector;
use crate::float::FftFloat;
use num_complex::Complex;

#[derive(Copy, Clone)]
pub struct Generic<T>(Complex<T>);

impl<T: FftFloat + Default> ComplexVector for Generic<T> {
    type Float = T;
    const WIDTH: usize = 1;

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(Complex::default())
    }

    #[inline(always)]
    unsafe fn broadcast(value: &Complex<Self::Float>) -> Self {
        Self(*value)
    }

    #[inline(always)]
    unsafe fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }

    #[inline(always)]
    unsafe fn sub(&self, rhs: &Self) -> Self {
        Self(self.0 - rhs.0)
    }

    #[inline(always)]
    unsafe fn mul(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    #[inline(always)]
    unsafe fn rotate(&self, positive: bool) -> Self {
        Self(if positive {
            Complex::new(-self.0.im, self.0.re)
        } else {
            Complex::new(self.0.im, -self.0.re)
        })
    }

    #[inline(always)]
    unsafe fn load(from: *const Complex<Self::Float>) -> Self {
        Self(*from)
    }

    #[inline(always)]
    unsafe fn store(&self, to: *mut Complex<Self::Float>) {
        *to = self.0;
    }

    #[inline(always)]
    unsafe fn partial_load(_from: *const Complex<Self::Float>, _count: usize) -> Self {
        unimplemented!("cannot partial load a generic vector")
    }

    #[inline(always)]
    unsafe fn partial_store(&self, _to: *mut Complex<Self::Float>, _count: usize) {
        unimplemented!("cannot partial store a generic vector")
    }
}
