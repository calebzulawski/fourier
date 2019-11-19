use super::BaseConfig;
use crate::float::FftFloat;
use crunchy::unroll;
use num_complex::Complex;

pub struct Radix4<T> {
    base: BaseConfig<T>,
    forward: bool,
}

impl<T: FftFloat> Radix4<T> {
    pub fn forward(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::forward(size, stride, 4),
            forward: true,
        }
    }

    pub fn inverse(size: usize, stride: usize) -> Self {
        Self {
            base: BaseConfig::inverse(size, stride, 4),
            forward: false,
        }
    }
}

#[inline]
pub fn radix4<T: FftFloat>(
    x: &[Complex<T>],
    y: &mut [Complex<T>],
    Radix4 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
        forward,
    }: &Radix4<T>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    #[inline]
    fn rotate<T: FftFloat>(z: Complex<T>, forward: bool) -> Complex<T> {
        if forward {
            Complex::new(-z.im, z.re)
        } else {
            Complex::new(z.im, -z.re)
        }
    }

    if *size == 4usize {
        for i in 0..*stride {
            let x0 = x[i];
            let x1 = x[i + stride];
            let x2 = x[i + 2 * stride];
            let x3 = x[i + 3 * stride];
            let y0 = x0 + x2;
            let y1 = x0 - x2;
            let y2 = x1 + x3;
            let y3 = rotate(x1 - x3, *forward);
            y[i] = y0 + y2;
            y[i + stride] = y1 - y3;
            y[i + 2 * stride] = y0 - y2;
            y[i + 3 * stride] = y1 + y3;
        }
    } else {
        let m = size / 4;
        for i in 0..m {
            let wi1 = twiddles[i];
            let wi2 = twiddles[i + m];
            let wi3 = twiddles[i + 2 * m];
            for j in 0..*stride {
                let x0 = x[j + stride * i];
                let x1 = x[j + stride * (i + m)];
                let x2 = x[j + stride * (i + 2 * m)];
                let x3 = x[j + stride * (i + 3 * m)];
                let y0 = x0 + x2;
                let y1 = x0 - x2;
                let y2 = x1 + x3;
                let y3 = rotate(x1 - x3, *forward);
                y[j + stride * (4 * i + 0)] = y0 + y2;
                y[j + stride * (4 * i + 1)] = (y1 - y3) * wi1;
                y[j + stride * (4 * i + 2)] = (y0 - y2) * wi2;
                y[j + stride * (4 * i + 3)] = (y1 + y3) * wi3;
            }
        }
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn radix4_f32_avx(
    x: &[Complex<f32>],
    y: &mut [Complex<f32>],
    Radix4 {
        base: BaseConfig {
            twiddles,
            stride,
            size,
        },
        forward,
    }: &Radix4<f32>,
) {
    assert_eq!(x.len(), size * stride);
    assert_eq!(y.len(), size * stride);
    assert!(*stride != 0);

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn rotate(z: __m256, forward: bool) -> __m256 {
        if forward {
            _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps(z, 0xb1))
        } else {
            _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), z), 0xb1)
        }
    }

    #[inline]
    unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        let a_re = _mm256_unpacklo_ps(a, a);
        let a_im = _mm256_unpackhi_ps(a, a);
        let b_sh = _mm256_shuffle_ps(b, b, 5);
        _mm256_addsub_ps(_mm256_mul_ps(a_re, b), _mm256_mul_ps(a_im, b_sh))
    }

    #[inline]
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

    let m = size / 4;
    let full_count = *stride / 4; // 4 values per register
    let partial_count = *stride - full_count * 4;
    for i in 0..m {
        // Load and broadcast twiddle factors
        let mut wi = [_mm256_setzero_ps(); 3];
        unroll! {
            for k in 0..3 {
                let twiddle = twiddles[i + k * m];
                wi[k] = _mm256_unpacklo_ps(_mm256_set1_ps(twiddle.re), _mm256_set1_ps(twiddle.im));
            }
        }

        // Loop over full vectors
        for j in 0..full_count {
            // Load full vectors
            let mut scratch = [_mm256_setzero_ps(); 4];
            let load = x.as_ptr().add(j + stride * i);
            unroll! {
                for k in 0..4 {
                    scratch[k] = _mm256_loadu_ps(load.add(stride * k * m) as *const f32);
                }
            }

            // Butterfly with optional twiddles
            scratch = bfly(scratch, *forward);
            if *size != 4 {
                unroll! {
                    for k in 0..3 {
                        scratch[k + 1] = mul(scratch[k + 1], wi[k]);
                    }
                }
            }

            // Store full vectors
            let store = y.as_ptr().add(j + 4 * stride * i);
            unroll! {
                for k in 0..4 {
                    _mm256_storeu_ps(store.add(stride * k) as *mut f32, scratch[k]);
                }
            }
        }

        // Apply the final partial vector
        if partial_count > 0 {
            // Load a partial vector
            let has_2 = if partial_count >= 2 { -1 } else { 0 };
            let has_3 = if partial_count >= 3 { -1 } else { 0 };
            let mask = _mm256_set_epi32(0, 0, has_3, has_3, has_2, has_2, -1, -1);
            let mut scratch = [_mm256_setzero_ps(); 4];
            let load = x.as_ptr().add(full_count + stride * i);
            unroll! {
                for k in 0..4 {
                    scratch[k] = _mm256_maskload_ps(load.add(stride * k * m) as *const f32, mask);
                }
            }

            // Butterfly with optional twiddles
            scratch = bfly(scratch, *forward);
            if *size != 4 {
                unroll! {
                    for k in 0..3 {
                        scratch[k + 1] = mul(scratch[k + 1], wi[k]);
                    }
                }
            }

            // Store a partial vector
            let store = y.as_ptr().add(full_count + 4 * stride * i);
            unroll! {
                for k in 0..4 {
                    _mm256_maskstore_ps(store.add(stride * k) as *mut f32, mask, scratch[k]);
                }
            }
        }
    }
}

pub fn radix4_f32(x: &[Complex<f32>], y: &mut [Complex<f32>], config: &Radix4<f32>) {
    unsafe { radix4_f32_avx(x, y, config) };
}
