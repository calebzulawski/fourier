use super::{radix2, radix4, BaseConfig};
use crate::float::FftFloat;
use crate::{avx, generic};
use num_complex::Complex;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct Radix8<T> {
    base: BaseConfig<T>,
    twiddle: Complex<T>,
}

impl<T: FftFloat> Radix8<T> {
    pub fn new(size: usize, stride: usize, forward: bool) -> Self {
        Self {
            base: BaseConfig::new(size, stride, 8, forward),
            twiddle: super::compute_twiddle(1, 8, forward),
        }
    }

    pub fn forward(size: usize, stride: usize) -> Self {
        Self::new(size, stride, true)
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self::new(size, stride, false)
    }
}

macro_rules! make_butterfly {
    {
        $input:ident,
        $forward:ident,
        $twiddle:ident,
        $twiddle_neg:ident,
        $bfly2:path,
        $bfly4:path,
        $mul:path,
        $rotate:path,
    } => {
        {
            let x = $input;
            let forward = $forward;
            let twiddle = $twiddle;
            let twiddle_neg = $twiddle_neg;
            let a1 = $bfly4([x[0], x[2], x[4], x[6]], forward);
            let mut b1 = $bfly4([x[1], x[3], x[5], x[7]], forward);
            b1[1] = $mul(b1[1], twiddle);
            b1[2] = $rotate(b1[2], !forward);
            b1[3] = $mul(b1[3], twiddle_neg);
            let a2 = $bfly2([a1[0], b1[0]], forward);
            let b2 = $bfly2([a1[1], b1[1]], forward);
            let c2 = $bfly2([a1[2], b1[2]], forward);
            let d2 = $bfly2([a1[3], b1[3]], forward);
            [a2[0], b2[0], c2[0], d2[0], a2[1], b2[1], c2[1], d2[1]]
        }
    }
}

#[inline(always)]
pub fn butterfly<T: FftFloat>(
    x: [Complex<T>; 8],
    forward: bool,
    twiddle: Complex<T>,
    twiddle_neg: Complex<T>,
) -> [Complex<T>; 8] {
    make_butterfly! {
        x,
        forward,
        twiddle,
        twiddle_neg,
        radix2::butterfly,
        radix4::butterfly,
        generic::mul,
        generic::rotate,
    }
}

#[multiversion::target_clones("[x86|x86_64]+avx", "[x86|x86_64]+avx+fma")]
#[inline]
pub unsafe fn butterfly_avx(
    x: [__m256; 8],
    forward: bool,
    twiddle: __m256,
    twiddle_neg: __m256,
) -> [__m256; 8] {
    #[static_dispatch]
    use avx::mul;
    make_butterfly! {
        x,
        forward,
        twiddle,
        twiddle_neg,
        radix2::butterfly_avx,
        radix4::butterfly_avx,
        mul,
        avx::rotate,
    }
}

#[inline]
pub fn radix8<T: FftFloat>(x: &[Complex<T>], y: &mut [Complex<T>], config: &Radix8<T>) {
    let twiddle = config.twiddle;
    let twiddle_neg = Complex::new(-config.twiddle.re, config.twiddle.im);

    let bfly = move |x: [Complex<T>; 8], forward: bool| -> [Complex<T>; 8] {
        butterfly(x, forward, twiddle, twiddle_neg)
    };

    crate::implement_generic! {8, x, y, &config.base, bfly}
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix8_f32_avx(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix8<f32>) {
    #[static_dispatch]
    use crate::avx::mul;
    #[static_dispatch]
    use butterfly_avx;

    let twiddle = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );
    let twiddle_neg = _mm256_blend_ps(
        _mm256_set1_ps(-config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );

    let bfly = move |x: [__m256; 8], forward: bool| -> [__m256; 8] {
        butterfly_avx(x, forward, twiddle, twiddle_neg)
    };

    crate::implement_avx_f32! {8, x, y, &config.base, bfly}
}

#[multiversion::target("[x86|x86_64]+avx+fma")]
unsafe fn radix8_f32_fma(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix8<f32>) {
    #[static_dispatch]
    use crate::avx::mul;

    let twiddle = _mm256_blend_ps(
        _mm256_set1_ps(config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );
    let twiddle_neg = _mm256_blend_ps(
        _mm256_set1_ps(-config.twiddle.re),
        _mm256_set1_ps(config.twiddle.im),
        0xaa,
    );

    let bfly = move |x: [__m256; 8], forward: bool| -> [__m256; 8] {
        butterfly_avx(x, forward, twiddle, twiddle_neg)
    };

    crate::implement_avx_f32! {8, x, y, &config.base, bfly}
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix8_f32_avx,
    "[x86|x86_64]+avx+fma" => radix8_f32_fma
)]
pub fn radix8_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix8<f32>) {
    radix8(x, y, config);
}
