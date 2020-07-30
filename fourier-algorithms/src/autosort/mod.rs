#![allow(unused_unsafe)]
#![allow(unused_macros)]

#[macro_use]
mod butterfly;
#[macro_use]
mod avx_optimization;

use crate::fft::{Fft, Transform};
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use core::cell::RefCell;
use core::marker::PhantomData;
use num_complex::Complex;
use num_traits::One as _;

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt without std

const NUM_RADICES: usize = 5;
const RADICES: [usize; NUM_RADICES] = [4, 8, 4, 3, 2];

/// Initializes twiddles.
fn initialize_twiddles<T: FftFloat, E: Extend<Complex<T>>>(
    mut size: usize,
    counts: [usize; NUM_RADICES],
    forward_twiddles: &mut E,
    inverse_twiddles: &mut E,
) {
    let mut stride = 1;
    for (radix, count) in RADICES.iter().zip(&counts) {
        for _ in 0..*count {
            let m = size / radix;
            for i in 0..m {
                forward_twiddles.extend(core::iter::once(Complex::<T>::one()));
                inverse_twiddles.extend(core::iter::once(Complex::<T>::one()));
                for j in 1..*radix {
                    forward_twiddles.extend(core::iter::once(compute_twiddle(i * j, size, true)));
                    inverse_twiddles.extend(core::iter::once(compute_twiddle(i * j, size, false)));
                }
            }
            size /= radix;
            stride *= radix;
        }
    }
}

/// Implements a mixed-radix Stockham autosort algorithm for multiples of 2 and 3.
pub struct Autosort<T, Twiddles, Work> {
    size: usize,
    counts: [usize; NUM_RADICES],
    forward_twiddles: Twiddles,
    inverse_twiddles: Twiddles,
    work: RefCell<Work>,
    real_type: PhantomData<T>,
}

impl<T, Twiddles, Work> Autosort<T, Twiddles, Work> {
    /// Return the radix counts.
    pub fn counts(&self) -> [usize; NUM_RADICES] {
        self.counts
    }

    /// Create a new transform generator from parts.  Twiddles factors and work must be the correct
    /// size.
    pub unsafe fn new_from_parts(
        size: usize,
        counts: [usize; NUM_RADICES],
        forward_twiddles: Twiddles,
        inverse_twiddles: Twiddles,
        work: Work,
    ) -> Self {
        Self {
            size,
            counts,
            forward_twiddles,
            inverse_twiddles,
            work: RefCell::new(work),
            real_type: PhantomData,
        }
    }
}

impl<T, Twiddles: AsRef<[Complex<T>]>, Work: AsRef<[Complex<T>]>> Autosort<T, Twiddles, Work> {
    /// Return the forward and inverse twiddle factors.
    pub fn twiddles(&self) -> (&[Complex<T>], &[Complex<T>]) {
        (
            self.forward_twiddles.as_ref(),
            self.inverse_twiddles.as_ref(),
        )
    }

    /// Return the work buffer size.
    pub fn work_size(&self) -> usize {
        self.work.borrow().as_ref().len()
    }
}

impl<T: FftFloat, Twiddles: Default + Extend<Complex<T>>, Work: Default + Extend<Complex<T>>>
    Autosort<T, Twiddles, Work>
{
    /// Create a new Stockham autosort generator.  Returns `None` if the transform size cannot be
    /// performed.
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
            let mut forward_twiddles = Twiddles::default();
            let mut inverse_twiddles = Twiddles::default();
            initialize_twiddles(size, counts, &mut forward_twiddles, &mut inverse_twiddles);
            let mut work = Work::default();
            work.extend(core::iter::repeat(Complex::default()).take(size));
            Some(Self {
                size,
                counts,
                forward_twiddles,
                inverse_twiddles,
                work: RefCell::new(work),
                real_type: PhantomData,
            })
        } else {
            None
        }
    }
}

macro_rules! implement {
    {
        $type:ty, $apply:ident
    } => {
        impl<Twiddles: AsRef<[Complex<$type>]>, Work: AsMut<[Complex<$type>]>> Fft
            for Autosort<$type, Twiddles, Work>
        {
            type Real = $type;

            fn size(&self) -> usize {
                self.size
            }

            fn transform_in_place(&self, input: &mut [Complex<$type>], transform: Transform) {
                let mut work = self.work.borrow_mut();
                let twiddles = if transform.is_forward() {
                    &self.forward_twiddles
                } else {
                    &self.inverse_twiddles
                };
                $apply(
                    input,
                    work.as_mut(),
                    &self.counts,
                    twiddles.as_ref(),
                    self.size,
                    transform,
                );
            }
        }
    }
}
implement! { f32, apply_stages_f32 }
implement! { f64, apply_stages_f64 }

/// This macro creates two modules, `radix_f32` and `radix_f64`, containing the radix application
/// functions for each radix.
macro_rules! make_radix_fns {
    {
        @impl $type:ty, $wide:literal, $radix:literal, $name:ident, $butterfly:ident
    } => {

        #[multiversion::multiversion]
        #[clone(target = "[x86|x86_64]+avx")]
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
                                broadcast!(cached_twiddles.as_ptr().add(i * $radix + k).read())
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
        #[multiversion::multiversion]
        #[clone(target = "[x86|x86_64]+avx")]
        #[inline]
        fn $name(
            input: &mut [Complex<$type>],
            output: &mut [Complex<$type>],
            stages: &[usize; NUM_RADICES],
            mut twiddles: &[Complex<$type>],
            mut size: usize,
            transform: Transform,
        ) {
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
                        8 => dispatch!($radix_mod::radix_8_narrow(from, to, transform.is_forward(), size, stride, twiddles)),
                        4 => dispatch!($radix_mod::radix_4_narrow(from, to, transform.is_forward(), size, stride, twiddles)),
                        3 => dispatch!($radix_mod::radix_3_narrow(from, to, transform.is_forward(), size, stride, twiddles)),
                        2 => dispatch!($radix_mod::radix_2_narrow(from, to, transform.is_forward(), size, stride, twiddles)),
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
                        8 => dispatch!($radix_mod::radix_8_wide(from, to, transform.is_forward(), size, stride, twiddles)),
                        4 => dispatch!($radix_mod::radix_4_wide(from, to, transform.is_forward(), size, stride, twiddles)),
                        3 => dispatch!($radix_mod::radix_3_wide(from, to, transform.is_forward(), size, stride, twiddles)),
                        2 => dispatch!($radix_mod::radix_2_wide(from, to, transform.is_forward(), size, stride, twiddles)),
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
