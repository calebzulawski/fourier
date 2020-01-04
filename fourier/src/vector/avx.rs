#![allow(unused_macros)]

#[macro_export]
#[doc(hidden)]
macro_rules! avx_vector {
    { f32 } => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        use num_complex::Complex;

        macro_rules! width {
            {} => { 4 }
        }

        macro_rules! zeroed {
            {} => { unsafe { _mm256_setzero_ps() } }
        }

        macro_rules! broadcast {
            { $z:expr } => {
                unsafe {
                    _mm256_blend_ps(
                        _mm256_set1_ps($z.re),
                        _mm256_set1_ps($z.im),
                        0xaa,
                    )
                }
            }
        }

        macro_rules! add {
            { $a:expr, $b:expr } => { unsafe { _mm256_add_ps($a,$b) } }
        }

        macro_rules! sub {
            { $a:expr, $b:expr } => { unsafe { _mm256_sub_ps($a, $b) } }
        }

        macro_rules! mul {
            { $a:expr, $b:expr } => {
                unsafe {
                    let re = _mm256_moveldup_ps($a);
                    let im = _mm256_movehdup_ps($a);
                    let sh = _mm256_permute_ps($b, 0xb1);
                    _mm256_addsub_ps(
                        _mm256_mul_ps(re, $b),
                        _mm256_mul_ps(im, sh),
                    )
                }
            }
        }

        macro_rules! rotate {
            { $z:expr, $positive:expr } => {
                unsafe {
                    if $positive {
                        _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps($z, 0xb1))
                    } else {
                        _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), $z), 0xb1)
                    }
                }
            }
        }

        macro_rules! load_wide {
            { $from:expr } => { _mm256_loadu_ps($from as *const f32) }
        }

        macro_rules! store_wide {
            { $z:expr, $to:expr } => { _mm256_storeu_ps($to as *mut f32, $z) }
        }

        macro_rules! load_narrow {
            { $from:expr } => {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    $from.read().im,
                    $from.read().re,
                )
            }
        }

        macro_rules! store_narrow {
            { $z:expr, $to:expr } => {
                $to.write(Complex::new(
                    _mm256_cvtss_f32($z),
                    _mm256_cvtss_f32(_mm256_permute_ps($z, 1)),
                ))
            }
        }
    };
    { f64 } => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        use num_complex::Complex;

        macro_rules! width {
            {} => { 2 }
        }

        macro_rules! zeroed {
            {} => { unsafe { _mm256_setzero_pd() } }
        }

        macro_rules! broadcast {
            { $z:expr } => {
                unsafe {
                    _mm256_blend_pd(
                        _mm256_set1_pd($z.re),
                        _mm256_set1_pd($z.im),
                        0xa,
                    )
                }
            }
        }

        macro_rules! add {
            { $a:expr, $b:expr } => { unsafe { _mm256_add_pd($a,$b) } }
        }

        macro_rules! sub {
            { $a:expr, $b:expr } => { unsafe { _mm256_sub_pd($a, $b) } }
        }

        macro_rules! mul {
            { $a:expr, $b:expr } => {
                unsafe {
                    let re = _mm256_unpacklo_pd($a, $a);
                    let im = _mm256_unpackhi_pd($a, $a);
                    let sh = _mm256_permute_pd($b, 0x5);
                    _mm256_addsub_pd(
                        _mm256_mul_pd(re, $b),
                        _mm256_mul_pd(im, sh),
                    )
                }
            }
        }

        macro_rules! rotate {
            { $z:expr, $positive:expr } => {
                unsafe {
                    if $positive {
                        _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_permute_pd($z, 0x5))
                    } else {
                        _mm256_permute_pd(_mm256_addsub_pd(_mm256_setzero_pd(), $z), 0x5)
                    }
                }
            }
        }

        macro_rules! load_wide {
            { $from:expr } => { _mm256_loadu_pd($from as *const f64) }
        }

        macro_rules! store_wide {
            { $z:expr, $to:expr } => { _mm256_storeu_pd($to as *mut f64, $z) }
        }

        macro_rules! load_narrow {
            { $from:expr } => {
                _mm256_set_pd(
                    0.0,
                    0.0,
                    $from.read().im,
                    $from.read().re,
                )
            }
        }

        macro_rules! store_narrow {
            { $z:expr, $to:expr } => {
                {
                    let lower = _mm256_extractf128_pd($z, 0);
                    $to.write(Complex::new(
                        _mm_cvtsd_f64(lower),
                        _mm_cvtsd_f64(_mm_permute_pd(lower, 1))
                    ));
                }
            }
        }
    }
}
