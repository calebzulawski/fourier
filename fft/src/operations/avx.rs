#[macro_export]
macro_rules! implement_avx_f32 {
    {
        $radix:literal,
        $input:expr,
        $output:expr,
        $config:expr,
        $butterfly:ident
    } => {
        {
            let BaseConfig::<f32> {
                twiddles,
                stride,
                size,
                forward,
            } = $config;

            assert_eq!($input.len(), size * stride);
            assert_eq!($output.len(), size * stride);
            assert!(*stride != 0);

            use crate::avx;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;

            let m = size / $radix;
            let full_count = (stride / 4) * 4; // 4 values per register
            let partial_count = stride - full_count;
            for i in 0..m {
                // Load and broadcast twiddle factors
                let mut wi = [_mm256_setzero_ps(); $radix - 1];
                for k in 0..($radix - 1) {
                    let twiddle = twiddles[i + k * m];
                    wi[k] = _mm256_blend_ps(_mm256_set1_ps(twiddle.re), _mm256_set1_ps(twiddle.im), 0xaa);
                }

                // Loop over full vectors
                for j in (0..full_count).step_by(4) {
                    // Load full vectors
                    let mut scratch = [_mm256_setzero_ps(); $radix];
                    let load = $input.as_ptr().add(j + stride * i);
                    for k in 0..$radix {
                        scratch[k] = _mm256_loadu_ps(load.add(stride * k * m) as *const f32);
                    }

                    // Butterfly with optional twiddles
                    scratch = $butterfly(scratch, *forward);
                    if *size != $radix {
                        for k in 0..($radix - 1) {
                            scratch[k + 1] = avx::mul(scratch[k + 1], wi[k]);
                        }
                    }

                    // Store full vectors
                    let store = $output.as_ptr().add(j + $radix * stride * i);
                    for k in 0..$radix {
                        _mm256_storeu_ps(store.add(stride * k) as *mut f32, scratch[k]);
                    }
                }

                // Apply the final partial vector
                if partial_count > 0 {
                    // Load a partial vector
                    let mask = avx::partial_mask(partial_count);
                    let mut scratch = [_mm256_setzero_ps(); $radix];
                    let load = $input.as_ptr().add(full_count + stride * i);
                    for k in 0..$radix {
                        scratch[k] = _mm256_maskload_ps(load.add(stride * k * m) as *const f32, mask);
                    }

                    // Butterfly with optional twiddles
                    scratch = $butterfly(scratch, *forward);
                    if *size != $radix {
                        for k in 0..($radix - 1) {
                            scratch[k + 1] = avx::mul(scratch[k + 1], wi[k]);
                        }
                    }

                    // Store a partial vector
                    let store = $output.as_ptr().add(full_count + $radix * stride * i);
                    for k in 0..$radix {
                        _mm256_maskstore_ps(store.add(stride * k) as *mut f32, mask, scratch[k]);
                    }
                }
            }
        }
    }
}
