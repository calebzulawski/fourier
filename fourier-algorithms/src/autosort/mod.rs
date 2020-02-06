#![allow(unused_unsafe)]
#![allow(unused_macros)]

#[macro_use]
mod butterfly;
#[macro_use]
mod avx_optimization;

use crate::fft::Transform;
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use num_complex::Complex;
use num_traits::One as _;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt without std

const NUM_RADICES: usize = 5;
const RADICES: [usize; NUM_RADICES] = [4, 8, 4, 3, 2];

pub struct Stages {
    size: usize,
    counts: [usize; NUM_RADICES],
}

impl Stages {
    pub fn new(size: usize) -> Option<Self> {
        let mut current_size = size;
        let mut counts = [0usize; NUM_RADICES];
        if current_size % RADICES[0] == 0 {
            current_size /= RADICES[0];
            counts[0] = 1;
        }
        for (count, radix) in counts.iter_mut().zip(&RADICES).skip(1) {
            while current_size % radix == 0 {
                current_size /= radix;
                *count += 1;
            }
        }
        if current_size == 1 {
            Some(Stages { size, counts })
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn initialize_twiddles<T: FftFloat, E: Extend<Complex<T>>>(
        &self,
        twiddles: &mut E,
        forward: bool,
    ) {
        let mut size = self.size;
        let mut stride = 1;
        for (radix, count) in RADICES.iter().zip(&self.counts) {
            for _ in 0..*count {
                let m = self.size / radix;
                for i in 0..m {
                    twiddles.extend(core::iter::once(Complex::<T>::one()));
                    for j in 1..*radix {
                        twiddles.extend(core::iter::once(compute_twiddle(
                            i * j,
                            self.size,
                            forward,
                        )));
                    }
                }
                size /= radix;
                stride *= radix;
            }
        }
    }

    pub fn apply_f32(
        &self,
        input: &mut [Complex<f32>],
        output: &mut [Complex<f32>],
        twiddles: &[Complex<f32>],
        transform: Transform,
    ) {
        apply_stages_f32(input, output, &self.counts, twiddles, self.size, transform);
    }

    pub fn apply_f64(
        &self,
        input: &mut [Complex<f64>],
        output: &mut [Complex<f64>],
        twiddles: &[Complex<f64>],
        transform: Transform,
    ) {
        apply_stages_f64(input, output, &self.counts, twiddles, self.size, transform);
    }
}

/// This macro creates two modules, `radix_f32` and `radix_f64`, containing the radix application
/// functions for each radix.
macro_rules! make_radix_fns {
    {
        @impl $type:ty, $wide:literal, $radix:literal, $name:ident, $butterfly:ident
    } => {

        #[multiversion::target_clones("[x86|x86_64]+avx")]
        #[inline]
        pub fn $name(
            input: &[num_complex::Complex<$type>],
            output: &mut [num_complex::Complex<$type>],
            _forward: bool,
            size: usize,
            stride: usize,
            cached_twiddles: &[num_complex::Complex<$type>],
        ) {
            #[target_cfg(target = "[x86|x86_64]+avx")]
            crate::avx_vector! { $type };

            #[target_cfg(not(target = "[x86|x86_64]+avx"))]
            crate::generic_vector! { $type };

            #[target_cfg(target = "[x86|x86_64]+avx")]
            {
                if !$wide && crate::avx_optimization!($type, $radix, input, output, _forward, size, stride, cached_twiddles) {
                    return
                }
            }

            let m = size / $radix;

            let (full_count, final_offset) = if $wide {
                (Some(((stride - 1) / width!()) * width!()), Some(stride - width!()))
            } else {
                (None, None)
            };

            for i in 0..m {
                // Load twiddle factors
                if $wide {
                    let twiddles = {
                        let mut twiddles = [zeroed!(); $radix];
                        for k in 1..$radix {
                            twiddles[k] = unsafe {
                                broadcast!(cached_twiddles.as_ptr().add((i * $radix + k)).read())
                            };
                        }
                        twiddles
                    };

                    // Loop over full vectors, with a final overlapping vector
                    for j in (0..full_count.unwrap())
                        .step_by(width!())
                        .chain(core::iter::once(final_offset.unwrap()))
                    {
                        // Load full vectors
                        let mut scratch = [zeroed!(); $radix];
                        let load = unsafe { input.as_ptr().add(j + stride * i) };
                        for k in 0..$radix {
                            scratch[k] = unsafe { load_wide!(load.add(stride * k * m)) };
                        }

                        // Butterfly with optional twiddles
                        scratch = $butterfly!($type, scratch, _forward);
                        if size != $radix {
                            for k in 1..$radix {
                                scratch[k] = mul!(scratch[k], twiddles[k]);
                            }
                        }

                        // Store full vectors
                        let store = unsafe { output.as_mut_ptr().add(j + $radix * stride * i) };
                        for k in 0..$radix {
                            unsafe { store_wide!(scratch[k], store.add(stride * k)) };
                        }
                    }
                } else {
                    let twiddles = {
                        let mut twiddles = [zeroed!(); $radix];
                        for k in 1..$radix {
                            twiddles[k] = unsafe {
                                load_narrow!(cached_twiddles.as_ptr().add(i * $radix + k))
                            };
                        }
                        twiddles
                    };

                    let load = unsafe { input.as_ptr().add(stride * i) };
                    let store = unsafe { output.as_mut_ptr().add($radix * stride * i) };
                    for j in 0..stride {
                        // Load a single value
                        let mut scratch = [zeroed!(); $radix];
                        for k in 0..$radix {
                            scratch[k] = unsafe { load_narrow!(load.add(stride * k * m + j)) };
                        }

                        // Butterfly with optional twiddles
                        scratch = $butterfly!($type, scratch, _forward);
                        if size != $radix {
                            for k in 1..$radix {
                                scratch[k] = mul!(scratch[k], twiddles[k]);
                            }
                        }

                        // Store a single value
                        for k in 0..$radix {
                            unsafe { store_narrow!(scratch[k], store.add(stride * k + j)) };
                        }
                    }
                }
            }
        }
    };
    {
        $([$radix:literal, $wide_name:ident, $narrow_name:ident, $butterfly:ident]),*
    } => {
        mod radix_f32 {
        $(
            make_radix_fns! { @impl f32, true, $radix, $wide_name, $butterfly }
            make_radix_fns! { @impl f32, false, $radix, $narrow_name, $butterfly }
        )*
        }
        mod radix_f64 {
        $(
            make_radix_fns! { @impl f64, true, $radix, $wide_name, $butterfly }
            make_radix_fns! { @impl f64, false, $radix, $narrow_name, $butterfly }
        )*
        }
    };
}

make_radix_fns! {
    [2, radix_2_wide, radix_2_narrow, butterfly2],
    [3, radix_3_wide, radix_3_narrow, butterfly3],
    [4, radix_4_wide, radix_4_narrow, butterfly4],
    [8, radix_8_wide, radix_8_narrow, butterfly8]
}

/// This macro creates the stage application function.
macro_rules! make_stage_fns {
    { $type:ty, $name:ident, $radix_mod:ident } => {
        #[multiversion::target_clones("[x86|x86_64]+avx")]
        #[inline]
        fn $name(
            input: &mut [Complex<$type>],
            output: &mut [Complex<$type>],
            stages: &[usize; NUM_RADICES],
            mut twiddles: &[Complex<$type>],
            mut size: usize,
            transform: Transform,
        ) {
            #[static_dispatch]
            use $radix_mod::radix_2_narrow;
            #[static_dispatch]
            use $radix_mod::radix_2_wide;
            #[static_dispatch]
            use $radix_mod::radix_3_narrow;
            #[static_dispatch]
            use $radix_mod::radix_3_wide;
            #[static_dispatch]
            use $radix_mod::radix_4_narrow;
            #[static_dispatch]
            use $radix_mod::radix_4_wide;
            #[static_dispatch]
            use $radix_mod::radix_8_narrow;
            #[static_dispatch]
            use $radix_mod::radix_8_wide;

            #[target_cfg(target = "[x86|x86_64]+avx")]
            crate::avx_vector! { $type };

            #[target_cfg(not(target = "[x86|x86_64]+avx"))]
            crate::generic_vector! { $type };

            assert_eq!(input.len(), output.len());
            assert_eq!(size, input.len());

            let mut stride = 1;

            let mut data_in_output = false;
            for (radix, iterations) in RADICES.iter().zip(stages) {
                let mut iteration = 0;

                // Use partial loads until the stride is large enough
                while stride < width! {} && iteration < *iterations {
                    let (from, to): (&mut _, &mut _) = if data_in_output {
                        (output, input)
                    } else {
                        (input, output)
                    };
                    match radix {
                        8 => radix_8_narrow(from, to, transform.is_forward(), size, stride, twiddles),
                        4 => radix_4_narrow(from, to, transform.is_forward(), size, stride, twiddles),
                        3 => radix_3_narrow(from, to, transform.is_forward(), size, stride, twiddles),
                        2 => radix_2_narrow(from, to, transform.is_forward(), size, stride, twiddles),
                        _ => unimplemented!("unsupported radix"),
                    }
                    size /= radix;
                    stride *= radix;
                    twiddles = &twiddles[size * radix..];
                    iteration += 1;
                    data_in_output = !data_in_output;
                }

                for _ in iteration..*iterations {
                    let (from, to): (&mut _, &mut _) = if data_in_output {
                        (output, input)
                    } else {
                        (input, output)
                    };
                    match radix {
                        8 => radix_8_wide(from, to, transform.is_forward(), size, stride, twiddles),
                        4 => radix_4_wide(from, to, transform.is_forward(), size, stride, twiddles),
                        3 => radix_3_wide(from, to, transform.is_forward(), size, stride, twiddles),
                        2 => radix_2_wide(from, to, transform.is_forward(), size, stride, twiddles),
                        _ => unimplemented!("unsupported radix"),
                    }
                    size /= radix;
                    stride *= radix;
                    twiddles = &twiddles[size * radix ..];
                    data_in_output = !data_in_output;
                }
            }
            if let Some(scale) = match transform {
                Transform::Fft | Transform::UnscaledIfft => None,
                Transform::Ifft => Some(1. / (input.len() as $type)),
                Transform::SqrtScaledFft | Transform::SqrtScaledIfft => Some(1. / (input.len() as $type).sqrt()),
            } {
                if data_in_output {
                    for (x, y) in output.iter().zip(input.iter_mut()) {
                        *y = x * scale;
                    }
                } else {
                    for x in input.iter_mut() {
                        *x *= scale;
                    }
                }
            } else {
                if data_in_output {
                    input.copy_from_slice(output);
                }
            }
        }
    };
}
make_stage_fns! { f32, apply_stages_f32, radix_f32 }
make_stage_fns! { f64, apply_stages_f64, radix_f64 }
