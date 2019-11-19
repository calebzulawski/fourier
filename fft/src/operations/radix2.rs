use super::BaseConfig;
use crate::float::FftFloat;
use num_complex::Complex;

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

#[inline]
pub fn radix2<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix2 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
    }: &Radix2<T>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    if *size == 2usize {
        for i in 0..*stride {
            let a = x[i];
            let b = x[i + stride];
            y[i] = a + b;
            y[i + stride] = a - b;
        }
    } else {
        let m = size / 2;
        for i in 0..m {
            let wi = twiddles[i];
            for j in 0..*stride {
                let a = x[j + stride * i];
                let b = x[j + stride * (i + m)];
                y[j + stride * 2 * i] = a + b;
                y[j + stride * (2 * i + 1)] = (a - b) * wi;
            }
        }
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix2_f32_avx(
    x: &[Complex<f32>],
    y: &mut [Complex<f32>],
    Radix2 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
    }: &Radix2<f32>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    use crate::avx;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn bfly(x: [__m256; 2]) -> [__m256; 2] {
        [_mm256_add_ps(x[0], x[1]), _mm256_sub_ps(x[0], x[1])]
    }

    let m = size / 2;
    let full_count = (*stride / 4) * 4; // 4 values per register
    let partial_count = *stride - full_count;
    for i in 0..m {
        // Load and broadcast twiddle factor
        let twiddle = twiddles[i];
        let wi = _mm256_blend_ps(_mm256_set1_ps(twiddle.re), _mm256_set1_ps(twiddle.im), 0xaa);

        // Loop over full vectors
        for j in (0..full_count).step_by(4) {
            // Load full vectors
            let load = x.as_ptr().add(j + stride * i);
            let mut scratch = [
                _mm256_loadu_ps(load as *const f32),
                _mm256_loadu_ps(load.add(stride * m) as *const f32),
            ];

            // Butterfly with optional twiddles
            scratch = bfly(scratch);
            if *size != 2 {
                scratch[1] = avx::mul(scratch[1], wi);
            }

            // Store full vectors
            let store = y.as_ptr().add(j + 4 * stride * i);
            _mm256_storeu_ps(store as *mut f32, scratch[0]);
            _mm256_storeu_ps(store.add(*stride) as *mut f32, scratch[1]);
        }

        // Apply the final partial vector
        if partial_count > 0 {
            // Load a partial vector
            let mask = avx::partial_mask(partial_count);
            let load = x.as_ptr().add(full_count + stride * i);
            let mut scratch = [
                _mm256_maskload_ps(load as *const f32, mask),
                _mm256_maskload_ps(load.add(stride * m) as *const f32, mask),
            ];

            // Butterfly with optional twiddles
            scratch = bfly(scratch);
            if *size != 2 {
                scratch[1] = avx::mul(scratch[1], wi);
            }

            // Store a partial vector
            let store = y.as_ptr().add(full_count + 4 * stride * i);
            _mm256_maskstore_ps(store as *mut f32, mask, scratch[0]);
            _mm256_maskstore_ps(store.add(*stride) as *mut f32, mask, scratch[1]);
        }
    }
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix2_f32_avx
)]
pub fn radix2_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix2<f32>) {
    radix2(x, y, config);
}
