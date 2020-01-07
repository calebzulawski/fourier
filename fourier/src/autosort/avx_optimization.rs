#![allow(unused_unsafe)]
#![allow(unused_macros)]

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
pub(crate) unsafe fn radix_4_stride_1_avx_f32(
    input: &[num_complex::Complex<f32>],
    output: &mut [num_complex::Complex<f32>],
    forward: bool,
    size: usize,
    twiddles: &[num_complex::Complex<f32>],
) {
    avx_vector! { f32 };
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
            out = mul!(out, twiddles);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(RADIX * i) as *mut _, out);
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! avx_optimization {
    {
        f32, $radix:literal, $input:ident, $output:ident, $forward:ident, $size:ident, $stride:ident, $twiddles:ident
    } => {
        if $radix == 4 && $stride == 1 {
            unsafe {
                crate::autosort::avx_optimization::radix_4_stride_1_avx_f32($input, $output, $forward, $size, $twiddles);
            }
            true
        } else {
            false
        }
    };
    {
        f64, $radix:literal, $input:ident, $output:ident, $forward:ident, $size:ident, $stride:ident, $twiddles:ident
    } => {
        // TODO f64 AVX init
        false
    };
    {
        $type:ty, $radix:literal, $input:ident, $output:ident, $forward:ident, $size:ident, $stride:ident, $twiddles:ident
    } => {
        false
    }
}
