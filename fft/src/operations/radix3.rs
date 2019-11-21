use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

pub struct Radix3<T> {
    base: BaseConfig<T>,
    twiddle: Complex<T>,
}

impl<T: FftFloat> Radix3<T> {
    fn new(size: usize, stride: usize, forward: bool) -> Self {
        Self {
            base: BaseConfig::new(size, stride, 3, forward),
            twiddle: super::compute_twiddle(1, 3, forward),
        }
    }

    pub fn forward(size: usize, stride: usize) -> Self {
        Self::new(size, stride, true)
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self::new(size, stride, false)
    }
}

#[inline]
pub fn radix3<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix3 {
        base: config,
        twiddle,
    }: &Radix3<T>,
) {
    let bfly = |x: [Complex<T>; 3], _forward: bool| -> [Complex<T>; 3] {
        [
            x[0] + x[1] + x[2],
            x[0] + x[1] * twiddle + x[2] * twiddle.conj(),
            x[0] + x[1] * twiddle.conj() + x[2] * twiddle,
        ]
    };

    crate::implement_generic! {3, x, y, config, bfly}
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix3_f32_avx(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix3<f32>) {
    #[static_dispatch]
    use crate::avx::mul;

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let twiddle = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );
    let twiddle_conj = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(-config.twiddle.im),
        0xaa,
    );

    let bfly = |x: [__m256; 3], _forward: bool| -> [__m256; 3] {
        [
            _mm256_add_ps(x[0], _mm256_add_ps(x[1], x[2])),
            _mm256_add_ps(
                x[0],
                _mm256_add_ps(mul(x[1], twiddle), mul(x[2], twiddle_conj)),
            ),
            _mm256_add_ps(
                x[0],
                _mm256_add_ps(mul(x[1], twiddle_conj), mul(x[2], twiddle)),
            ),
        ]
    };

    crate::implement_avx_f32! {3, x, y, &config.base, bfly}
}

#[multiversion::target("[x86|x86_64]+avx+fma")]
unsafe fn radix3_f32_fma(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix3<f32>) {
    #[static_dispatch]
    use crate::avx::mul;

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let twiddle = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );
    let twiddle_conj = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(-config.twiddle.im),
        0xaa,
    );

    let bfly = |x: [__m256; 3], _forward: bool| -> [__m256; 3] {
        [
            _mm256_add_ps(x[0], _mm256_add_ps(x[1], x[2])),
            _mm256_add_ps(
                x[0],
                _mm256_add_ps(mul(x[1], twiddle), mul(x[2], twiddle_conj)),
            ),
            _mm256_add_ps(
                x[0],
                _mm256_add_ps(mul(x[1], twiddle_conj), mul(x[2], twiddle)),
            ),
        ]
    };

    crate::implement_avx_f32! {3, x, y, &config.base, bfly}
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix3_f32_avx,
    "[x86|x86_64]+avx+fma" => radix3_f32_fma
)]
pub fn radix3_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix3<f32>) {
    radix3(x, y, config);
}
