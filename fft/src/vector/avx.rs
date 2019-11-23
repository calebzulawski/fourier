use num_complex::Complex;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Copy, Clone)]
pub struct Avx32(__m256);

impl super::ComplexVector for Avx32 {
    type Float = f32;
    const WIDTH: usize = 4;

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn zero() -> Self {
        Self(_mm256_setzero_ps())
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn broadcast(value: &Complex<Self::Float>) -> Self {
        Self(_mm256_blend_ps(
            _mm256_set1_ps(value.re),
            _mm256_set1_ps(value.im),
            0xaa,
        ))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn add(&self, rhs: &Self) -> Self {
        Self(_mm256_add_ps(self.0, rhs.0))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn sub(&self, rhs: &Self) -> Self {
        Self(_mm256_sub_ps(self.0, rhs.0))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn mul(&self, rhs: &Self) -> Self {
        let re = _mm256_moveldup_ps(self.0);
        let im = _mm256_movehdup_ps(self.0);
        let sh = _mm256_permute_ps(rhs.0, 0xb1);
        Self(_mm256_addsub_ps(
            _mm256_mul_ps(re, rhs.0),
            _mm256_mul_ps(im, sh),
        ))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn rotate(&self, positive: bool) -> Self {
        Self(if positive {
            _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps(self.0, 0xb1))
        } else {
            _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), self.0), 0xb1)
        })
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn load(from: *const Complex<Self::Float>) -> Self {
        Self(_mm256_loadu_ps(from as *const Self::Float))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn store(&self, to: *mut Complex<Self::Float>) {
        _mm256_storeu_ps(to as *mut Self::Float, self.0);
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn partial_load(from: *const Complex<Self::Float>, count: usize) -> Self {
        assert!(count < 4);
        assert!(count > 0);
        let has_2 = if count >= 2 { -1 } else { 0 };
        let has_3 = if count >= 3 { -1 } else { 0 };
        let mask = _mm256_set_epi32(0, 0, has_3, has_3, has_2, has_2, -1, -1);
        Self(_mm256_maskload_ps(from as *const f32, mask))
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[inline]
    unsafe fn partial_store(&self, to: *mut Complex<Self::Float>, count: usize) {
        assert!(count < 4);
        assert!(count > 0);
        let has_2 = if count >= 2 { -1 } else { 0 };
        let has_3 = if count >= 3 { -1 } else { 0 };
        let mask = _mm256_set_epi32(0, 0, has_3, has_3, has_2, has_2, -1, -1);
        _mm256_maskstore_ps(to as *mut f32, mask, self.0);
    }
}
