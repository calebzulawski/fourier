use super::{radix2, BaseConfig};
use crate::float::FftFloat;
use crate::{avx, generic};
use num_complex::Complex;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

macro_rules! make_butterfly {
    {
        $input:ident, $forward:ident, $bfly2:path, $rotate:path
    } => {
        {
            let x = $input;
            let forward = $forward;
            let a1 = $bfly2([x[0], x[2]], forward);
            let mut b1 = $bfly2([x[1], x[3]], forward);
            b1[1] = $rotate(b1[1], forward);
            let a2 = $bfly2([a1[0], b1[0]], forward);
            let b2 = $bfly2([a1[1], b1[1]], forward);
            [a2[0], b2[1], a2[1], b2[0]]
        }
    }
}

#[inline(always)]
pub fn butterfly<T: FftFloat>(x: [Complex<T>; 4], forward: bool) -> [Complex<T>; 4] {
    make_butterfly!(x, forward, radix2::butterfly, generic::rotate)
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub unsafe fn butterfly_avx(x: [__m256; 4], forward: bool) -> [__m256; 4] {
    make_butterfly!(x, forward, radix2::butterfly_avx, avx::rotate)
}

#[inline]
pub fn radix4<T: FftFloat>(x: &[Complex<T>], y: &mut [Complex<T>], config: &Radix4<T>) {
    crate::implement_generic! {4, x, y, &config.base, butterfly}
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix4_f32_avx(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    #[static_dispatch]
    use crate::avx::mul;
    crate::implement_avx_f32! {4, x, y, &config.base, butterfly_avx}
}

#[multiversion::target("[x86|x86_64]+avx+fma")]
unsafe fn radix4_f32_fma(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    #[static_dispatch]
    use crate::avx::mul;
    crate::implement_avx_f32! {4, x, y, &config.base, butterfly_avx}
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix4_f32_avx,
    "[x86|x86_64]+avx+fma" => radix4_f32_fma
)]
pub fn radix4_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    radix4(x, y, config);
}
