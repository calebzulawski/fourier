use num_complex::Complex;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sin/cos without std

#[inline]
pub fn compute_twiddle<T: num_traits::Num>(index: usize, size: usize, forward: bool) -> Complex<T> {
    let theta = (index * 2) as f64 * core::f64::consts::PI / size as f64;
    let twiddle = Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    );
    if forward {
        twiddle
    } else {
        twiddle.conj()
    }
}
