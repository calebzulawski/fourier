use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct Radix2<T> {
    base: BaseConfig<T>,
}

impl<T: FftFloat> Radix2<T> {
    pub fn forward(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::forward(size, stride, 2),
        }
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::inverse(size, stride, 2),
        }
    }
}

#[inline(always)]
pub fn butterfly<T: FftFloat>(x: [Complex<T>; 2], _forward: bool) -> [Complex<T>; 2] {
    [x[0] + x[1], x[0] - x[1]]
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub unsafe fn butterfly_avx(x: [__m256; 2], _forward: bool) -> [__m256; 2] {
    [_mm256_add_ps(x[0], x[1]), _mm256_sub_ps(x[0], x[1])]
}

#[inline]
pub fn radix2<T: FftFloat>(x: &[Complex<T>], y: &mut [Complex<T>], config: &Radix2<T>) {
    crate::implement_generic! {2, x, y, &config.base, butterfly}
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix2_f32_avx(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix2<f32>) {
    crate::implement_avx_f32! {2, x, y, &config.base, butterfly_avx}
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix2_f32_avx
)]
pub fn radix2_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix2<f32>) {
    radix2(x, y, config);
}
