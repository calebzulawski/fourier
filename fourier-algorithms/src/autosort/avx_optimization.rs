#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn cmul_f32(a: __m256, b: __m256) -> __m256 {
    let re = _mm256_moveldup_ps(a);
    let im = _mm256_movehdup_ps(a);
    let sh = _mm256_shuffle_ps(b, b, 0xb1);
    _mm256_addsub_ps(_mm256_mul_ps(re, b), _mm256_mul_ps(im, sh))
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn cmul_f64(a: __m256d, b: __m256d) -> __m256d {
    let re = _mm256_shuffle_pd(a, a, 0x00);
    let im = _mm256_shuffle_pd(a, a, 0x03);
    let sh = _mm256_shuffle_pd(b, b, 0x01);
    _mm256_addsub_pd(_mm256_mul_pd(re, b), _mm256_mul_pd(im, sh))
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub(crate) unsafe fn radix_4_stride_1_avx_f32(
    input: &[num_complex::Complex<f32>],
    output: &mut [num_complex::Complex<f32>],
    size: usize,
    stride: usize,
    twiddles: &[num_complex::Complex<f32>],
    forward: bool,
) {
    assert_eq!(stride, 1);
    const RADIX: usize = 4;
    let m = size / RADIX;

    for i in 0..m {
        // Load
        let gathered = _mm256_set_ps(
            input.as_ptr().add(3 * m + i).read().im,
            input.as_ptr().add(3 * m + i).read().re,
            input.as_ptr().add(2 * m + i).read().im,
            input.as_ptr().add(2 * m + i).read().re,
            input.as_ptr().add(m + i).read().im,
            input.as_ptr().add(m + i).read().re,
            input.as_ptr().add(i).read().im,
            input.as_ptr().add(i).read().re,
        );

        // first radix 2
        // in vector      |  sorted
        // ar1 = r0 - r2  |  ar0 = r0 + r2
        // ar0 = r0 + r2  |  ai0 = i0 + i2
        // ai1 = i0 - i2  |  ar1 = r0 - r2
        // ai0 = i0 + i2  |  ai1 = i0 - i2
        // ar3 = r1 - r3  |  ar2 = r1 + r3
        // ar2 = r1 + r3  |  ai2 = i1 + i3
        // ai3 = i1 - i3  |  ar3 = r1 - r3
        // ai2 = i1 + i3  |  ai3 = i1 - i3
        let a_lo = _mm256_unpacklo_ps(gathered, gathered); // r0 r0 i0 i0 r2 r2 i2 i2
        let a_hi = _mm256_unpackhi_ps(gathered, gathered); // r1 r1 i1 i1 r3 r3 i3 i3
        let a_jumbled = _mm256_addsub_ps(
            _mm256_permute2f128_ps(a_lo, a_hi, 0x20),
            _mm256_permute2f128_ps(a_lo, a_hi, 0x31),
        );

        // rotate
        // swap ar3 and ai3, conditionally negate one
        let a_swapped = _mm256_blend_ps(
            a_jumbled,
            _mm256_permute_ps(a_jumbled, 0b01_00_11_10),
            0b0101_0000,
        );

        let a_negated = _mm256_sub_ps(_mm256_setzero_ps(), a_swapped);
        let a_rotated = if forward {
            _mm256_blend_ps(a_swapped, a_negated, 0b0001_0000) // negate new ar3
        } else {
            _mm256_blend_ps(a_swapped, a_negated, 0b0100_0000) // negate new ai3
        };

        // second radix 2
        // in vector      |  sorted
        // br1 ar0 - ar2  |  br0 ar0 + ar2
        // br0 ar0 + ar2  |  bi0 ai0 + ai2
        // bi1 ai0 - ai2  |  br1 ar0 - ar2
        // bi0 ai0 + ai2  |  bi1 ai0 - ai2
        // br3 ar1 - ar3  |  br2 ar1 + ar3
        // br2 ar1 + ar3  |  bi2 ai1 + ai3
        // bi3 ai1 - ai3  |  br3 ar1 - ar3
        // bi2 ai1 + ai3  |  bi3 ai1 - ai3
        let b_lo = _mm256_permute_ps(a_rotated, 0b11_11_01_01); // ar0 ar0 ai0 ai0 ar2 ar2 ai2 ai2
        let b_hi = _mm256_permute_ps(a_rotated, 0b10_10_00_00); // ar1 ar1 ai1 ai1 ar3 ar3 ai3 ai3
        let b_jumbled = _mm256_addsub_ps(
            _mm256_permute2f128_ps(b_lo, b_hi, 0x20), // ar0 ar0 ai0 ai0 ar1 ar1 ai1 ai1
            _mm256_permute2f128_ps(b_lo, b_hi, 0x31), // ar2 ar2 ai2 ai2 ar3 ar3 ai3 ai3
        );

        // output br0 bi0 br3 bi3 br1 bi1 br2 bi2
        let out_lo = _mm256_permute_ps(b_jumbled, 0b11_01_11_01); // br0 bi0 br0 bi0 br2 bi2 br2 bi2
        let out_hi = {
            let temp = _mm256_permute_ps(b_jumbled, 0b10_00_10_00);
            _mm256_permute2f128_ps(temp, temp, 0x01)
        }; // br3 bi3 br3 bi3 br1 bi1 br1 bi1
        let mut out = _mm256_blend_ps(out_lo, out_hi, 0b0011_1100); // br0 bi0 br3 bi3 br1 bi1 br2 bi2
        if size != RADIX {
            let twiddles = _mm256_loadu_ps(twiddles.as_ptr().add(RADIX * i) as *const _);
            out = cmul_f32(out, twiddles);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(RADIX * i) as *mut _, out);
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn radix_4_stride_1_avx_f64(
    input: &[num_complex::Complex<f64>],
    output: &mut [num_complex::Complex<f64>],
    forward: bool,
    size: usize,
    twiddles: &[num_complex::Complex<f64>],
) {
    const RADIX: usize = 4;
    let m = size / RADIX;

    for i in 0..m {
        // Load
        let gathered2 = _mm256_set_pd(
            input.as_ptr().add(3 * m + i).read().im,
            input.as_ptr().add(3 * m + i).read().re,
            input.as_ptr().add(2 * m + i).read().im,
            input.as_ptr().add(2 * m + i).read().re,
        );
        let gathered1 = _mm256_set_pd(
            input.as_ptr().add(m + i).read().im,
            input.as_ptr().add(m + i).read().re,
            input.as_ptr().add(i).read().im,
            input.as_ptr().add(i).read().re,
        );

        // first radix 2
        // in vector      |  sorted
        // ar1 = r0 - r2  |  ar0 = r0 + r2
        // ar0 = r0 + r2  |  ai0 = i0 + i2
        // ar3 = r1 - r3  |  ar1 = r0 - r2
        // ar2 = r1 + r3  |  ai1 = i0 - i2
        // ---------------+---------------
        // ai1 = i0 - i2  |  ar2 = r1 + r3
        // ai0 = i0 + i2  |  ai2 = i1 + i3
        // ai3 = i1 - i3  |  ar3 = r1 - r3
        // ai2 = i1 + i3  |  ai3 = i1 - i3
        let a_lo1 = _mm256_unpacklo_pd(gathered1, gathered1); // r0 r0 r1 r1
        let a_lo2 = _mm256_unpacklo_pd(gathered2, gathered2); // r2 r2 r3 r3
        let a_hi1 = _mm256_unpackhi_pd(gathered1, gathered1); // i0 i0 i1 i1
        let a_hi2 = _mm256_unpackhi_pd(gathered2, gathered2); // i2 i2 i3 i3
        let a_jumbled1 = _mm256_addsub_pd(a_lo1, a_lo2);
        let a_jumbled2 = _mm256_addsub_pd(a_hi1, a_hi2);

        // rotate
        // swap ar3 and ai3, conditionally negate one
        let a_swapped1 = _mm256_blend_pd(a_jumbled1, a_jumbled2, 0b0100);
        let a_swapped2 = _mm256_blend_pd(a_jumbled2, a_jumbled1, 0b0100);

        let (a_rotated1, a_rotated2) = if forward {
            let a_negated1 = _mm256_sub_pd(_mm256_setzero_pd(), a_swapped1);
            (_mm256_blend_pd(a_swapped1, a_negated1, 0b0100), a_swapped2) // negate new ar3
        } else {
            let a_negated2 = _mm256_sub_pd(_mm256_setzero_pd(), a_swapped2);
            (a_swapped1, _mm256_blend_pd(a_swapped2, a_negated2, 0b0100)) // negate new ai3
        };

        // second radix 2
        // in vector      |  sorted
        // br1 ar0 - ar2  |  br0 ar0 + ar2
        // br0 ar0 + ar2  |  bi0 ai0 + ai2
        // bi1 ai0 - ai2  |  br1 ar0 - ar2
        // bi0 ai0 + ai2  |  bi1 ai0 - ai2
        // ---------------+---------------
        // br3 ar1 - ar3  |  br2 ar1 + ar3
        // br2 ar1 + ar3  |  bi2 ai1 + ai3
        // bi3 ai1 - ai3  |  br3 ar1 - ar3
        // bi2 ai1 + ai3  |  bi3 ai1 - ai3
        let b_lo = _mm256_permute2f128_pd(a_rotated1, a_rotated2, 0x20); // ar1 ar0 ai1 ai0
        let b_hi = _mm256_permute2f128_pd(a_rotated1, a_rotated2, 0x31); // ar3 ar2 ai3 ai2
        let b_jumbled1 = _mm256_addsub_pd(
            _mm256_unpackhi_pd(b_lo, b_lo), // ar0 ar0 ai0 ai0
            _mm256_unpackhi_pd(b_hi, b_hi), // ar2 ar2 ai2 ai2
        );
        let b_jumbled2 = _mm256_addsub_pd(
            _mm256_unpacklo_pd(b_lo, b_lo), // ar1 ar1 ai1 ai1
            _mm256_unpacklo_pd(b_hi, b_hi), // ar3 ar3 ai3 ai3
        );

        // output br0 bi0 br3 bi3 br1 bi1 br2 bi2
        let out_real = _mm256_permute2f128_pd(b_jumbled1, b_jumbled2, 0x20); // br1 br0 br3 br2
        let out_imag = _mm256_permute2f128_pd(b_jumbled1, b_jumbled2, 0x31); // bi1 bi0 bi3 bi2
        let mut out1 = _mm256_blend_pd(
            _mm256_permute_pd(out_real, 0b0011), // br0 br0 br3 br3
            _mm256_permute_pd(out_imag, 0b0011), // bi0 bi0 bi3 bi3
            0b1010,
        );
        let mut out2 = _mm256_blend_pd(
            _mm256_permute_pd(out_real, 0b1100), // br1 br1 br2 br2
            _mm256_permute_pd(out_imag, 0b1100), // bi1 bi1 bi2 bi2
            0b1010,
        );
        if size != RADIX {
            let twiddles1 = _mm256_loadu_pd(twiddles.as_ptr().add(RADIX * i) as *const _);
            let twiddles2 = _mm256_loadu_pd(twiddles.as_ptr().add(RADIX * i + 2) as *const _);
            out1 = cmul_f64(out1, twiddles1);
            out2 = cmul_f64(out2, twiddles2);
        }
        _mm256_storeu_pd(output.as_mut_ptr().add(RADIX * i) as *mut _, out1);
        _mm256_storeu_pd(output.as_mut_ptr().add(RADIX * i + 2) as *mut _, out2);
    }
}
