use crate::float::FftFloat;
use num_complex::Complex;

#[inline]
pub fn mul<T: FftFloat>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    a * b
}

#[inline]
pub fn rotate<T: FftFloat>(z: Complex<T>, forward: bool) -> Complex<T> {
    if forward {
        Complex::new(-z.im, z.re)
    } else {
        Complex::new(z.im, -z.re)
    }
}
