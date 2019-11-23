use crate::float::FftFloat;
use num_complex::Complex;

pub fn compute_twiddle<T: FftFloat>(index: usize, size: usize, forward: bool) -> Complex<T> {
    let theta = (index * 2) as f64 * std::f64::consts::PI / size as f64;
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
