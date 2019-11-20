use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Radix4<T> {
    base: BaseConfig<T>,
}

impl<T: FftFloat> Radix4<T> {
    pub fn forward(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::forward(size, stride, 4),
        }
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::inverse(size, stride, 4),
        }
    }
}

#[inline]
pub fn radix4<T: FftFloat>(x: &[Complex<T>], y: &mut [Complex<T>], config: &Radix4<T>) {
    #[inline]
    fn rotate<T: FftFloat>(z: Complex<T>, forward: bool) -> Complex<T> {
        if forward {
            Complex::new(-z.im, z.re)
        } else {
            Complex::new(z.im, -z.re)
        }
    }

    #[inline(always)]
    fn bfly<T: FftFloat>(x: [Complex<T>; 4], forward: bool) -> [Complex<T>; 4] {
        [
            x[0] + x[2],
            x[0] - x[2],
            x[1] + x[3],
            rotate(x[1] - x[3], forward),
        ]
    }

    crate::implement_generic! {4, x, y, &config.base, bfly}
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix4_f32_avx(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[multiversion::target("[x86|x86_64]+avx")]
    unsafe fn rotate(z: __m256, forward: bool) -> __m256 {
        if forward {
            _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps(z, 0xb1))
        } else {
            _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), z), 0xb1)
        }
    }

    #[inline(always)]
    unsafe fn bfly(x: [__m256; 4], forward: bool) -> [__m256; 4] {
        let i0 = _mm256_add_ps(x[0], x[2]);
        let i1 = _mm256_sub_ps(x[0], x[2]);
        let i2 = _mm256_add_ps(x[1], x[3]);
        let i3 = rotate(_mm256_sub_ps(x[1], x[3]), forward);
        let y0 = _mm256_add_ps(i0, i2);
        let y1 = _mm256_sub_ps(i1, i3);
        let y2 = _mm256_sub_ps(i0, i2);
        let y3 = _mm256_add_ps(i1, i3);
        [y0, y1, y2, y3]
    }

    crate::implement_avx_f32! {4, x, y, &config.base, bfly}
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix4_f32_avx
)]
pub fn radix4_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    radix4(x, y, config);
}
