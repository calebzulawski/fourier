macro_rules! generic_vector {
    {} => {
        macro_rules! add {
            { $a:expr, $b:expr } => { ($a + $b) }
        }

        macro_rules! sub {
            { $a:expr, $b:expr } => { ($a - $b) }
        }

        macro_rules! mul {
            { $a:expr, $b:expr } => { ($a * $b) }
        }

        macro_rules! broadcast {
            { $z:expr } => { ($z) }
        }

        macro_rules! rotate {
            { $positive:expr, $z:expr } => {
                if positive {
                    complex::new(-z.im, z.re)
                } else {
                    complex::new(z.im, -z.re)
                }
            }
        }

        macro_rules! load {
            { $from:expr } => { unsafe { *from } }
        }

        macro_rules! store {
            { $z:expr, $to:expr } => { unsafe { *to } = z }
        }
    }
}

macro_rules! avx_vector {
    { packed } => {
        avx_vector! { @common }

        macro_rules! load {
            { $from:expr } => { unsafe { _mm256_loadu_ps($from as *const f32) } }
        }

        macro_rules! store {
            { $z:expr, $to:expr } => { unsafe { _mm256_storeu_ps($to as *mut f32, $z) } }
        }
    },
    { single } => {
        avx_vector! { @common }

        macro_rules! load {
            { $from:expr } => {
                unsafe { _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    $from.read().im,
                    $from.read().re,
                )}
            }
        }

        macro_rules! store {
            { $z:expr, $to:expr } => {
                unsafe { to.write(Complex::new(
                    _mm256_cvtss_f32(z),
                    _mm256_cvtss_f32(_mm256_permute_ps(z, 1)),
                ))}
            }
        }
    },
    { @common } => {
        #[cfg(target_arch = "x86")] use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")] use core::arch::x86_64::*;

        macro_rules! add {
            { $a:expr, $b:expr } => { unsafe { _mm256_add_ps($a, $b) } }
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

        macro_rules! rotate {
            { $positive:expr, $z:expr } => {
                unsafe {
                    if $positive {
                        _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_permute_ps($z, 0xb1))
                    } else {
                        _mm256_permute_ps(_mm256_addsub_ps(_mm256_setzero_ps(), $z), 0xb1)
                    }
                }
            }
        }
    }
}

macro_rules! radix {
    {2, $forward, $x:expr} => { (add!($x.0, $x.1), sub!($x.0, $x.1)) },
    {4, $forward, $x:expr} => {
        {
            let a1 = radix!(2, forward, ($x.0, $x.2));
            let mut b1 = radix!(2, $forward, ($x.1, $x.3));
            b1.1 = rotate!($forward, b1.1);
            let a2 = radix!(2, $forward, (a1.0, b1.0));
            let b2 = radix!(2, $forward, (a1.1, b1.1));
            (a2.0, b2.1, a2.1, b2.0)
        }
    }
}
