#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub unsafe fn rotate(z: __m256, forward: bool) -> __m256 {
    if forward {
        _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps(z, 0xb1))
    } else {
        _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), z), 0xb1)
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub unsafe fn mul(a: __m256, b: __m256) -> __m256 {
    let a_re = _mm256_moveldup_ps(a);
    let a_im = _mm256_movehdup_ps(a);
    let b_sh = _mm256_permute_ps(b, 0xb1);
    _mm256_addsub_ps(_mm256_mul_ps(a_re, b), _mm256_mul_ps(a_im, b_sh))
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub unsafe fn partial_mask(count: usize) -> __m256i {
    assert!(count < 4);
    assert!(count > 0);
    let has_2 = if count >= 2 { -1 } else { 0 };
    let has_3 = if count >= 3 { -1 } else { 0 };
    _mm256_set_epi32(0, 0, has_3, has_3, has_2, has_2, -1, -1)
}
