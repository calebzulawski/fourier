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
use safe_simd::vector::{Feature, VectorCore};

#[cfg(not(feature = "std"))]
use num_traits::Float as _; // enable sqrt without std

#[derive(Copy, Clone, Default)]
pub struct StepParameters {
    pub size: usize,
    pub radix: usize,
    pub stride: usize,
}

impl StepParameters {
    fn initialize_twiddles<T: FftFloat>(
        &self,
        forward: &mut [Complex<T>],
        inverse: &mut [Complex<T>],
    ) {
        let m = self.size / self.radix;
        for i in 0..m {
            forward[0] = Complex::one();
            inverse[0] = Complex::one();
            for j in 1..self.radix {
                forward[j] = compute_twiddle(i * j, self.size, true);
                inverse[j] = compute_twiddle(i * j, self.size, false);
            }
        }
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
pub fn steps(size: usize) -> Option<Vec<StepParameters>> {
    let mut steps = Vec::new();
    let mut current_size = size;
    let mut stride = 1;

    // First step is radix 4 (helps performance)
    if current_size % 4 == 0 {
        steps.push(StepParameters {
            size: current_size,
            radix: 4,
            stride,
        });
        current_size /= 4;
        stride *= 4;
    }

    for radix in [8, 4, 3, 2].iter().copied() {
        while current_size % radix == 0 {
            steps.push(StepParameters {
                size: current_size,
                radix,
                stride,
            });
            current_size /= radix;
            stride *= radix;
        }
    }
    if current_size == 1 {
        Some(steps)
    } else {
        None
    }
}

pub fn num_twiddles(steps: &[StepParameters]) -> usize {
    let mut count = 0;
    for step in steps {
        count += step.size;
    }
    count
}

type StepFn<T> = unsafe fn(
    &[num_complex::Complex<T>],
    &mut [num_complex::Complex<T>],
    usize,
    usize,
    &[num_complex::Complex<T>],
    bool,
);

pub struct Step<T> {
    parameters: StepParameters,
    func: StepFn<T>,
    twiddle_offset: usize,
}

impl<T> Default for Step<T> {
    fn default() -> Self {
        Self {
            parameters: Default::default(),
            func: |_, _, _, _, _, _| panic!("uninitialized step!"),
            twiddle_offset: 0,
        }
    }
}

impl<T> Step<T> {
    fn apply(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        twiddles: &[Complex<T>],
        forward: bool,
    ) {
        let twiddles = &twiddles[self.twiddle_offset..];
        unsafe {
            (self.func)(
                input,
                output,
                self.parameters.size,
                self.parameters.stride,
                twiddles,
                forward,
            );
        }
    }
}

impl Step<f32> {
    fn new(parameters: StepParameters, twiddle_offset: usize) -> Self {
        assert!(parameters.size % parameters.radix == 0);

        // AVX wide implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if parameters.stride <= safe_simd::x86::avx::Vcf32::width()
                && safe_simd::x86::avx::Avx::new().is_some()
            {
                let func: StepFn<f32> = match parameters.radix {
                    2 => butterfly::radix2_wide_f32_avx_version,
                    3 => butterfly::radix3_wide_f32_avx_version,
                    4 => butterfly::radix4_wide_f32_avx_version,
                    8 => butterfly::radix8_wide_f32_avx_version,
                    radix => panic!("invalid radix: {}", radix),
                };
                return Self {
                    parameters,
                    func,
                    twiddle_offset,
                };
            };
        }

        // SSE wide implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if parameters.stride <= safe_simd::x86::sse::Vcf32::width()
                && safe_simd::x86::sse::Sse::new().is_some()
            {
                let func: StepFn<f32> = match parameters.radix {
                    2 => butterfly::radix2_wide_f32_sse3_version,
                    3 => butterfly::radix3_wide_f32_sse3_version,
                    4 => butterfly::radix4_wide_f32_sse3_version,
                    8 => butterfly::radix8_wide_f32_sse3_version,
                    radix => panic!("invalid radix: {}", radix),
                };
                return Self {
                    parameters,
                    func,
                    twiddle_offset,
                };
            };
        }

        // AVX narrow implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if safe_simd::x86::avx::Avx::new().is_some() {
                let func: StepFn<f32> = match parameters.radix {
                    2 => butterfly::radix2_narrow_f32_avx_version,
                    3 => butterfly::radix3_narrow_f32_avx_version,
                    4 => butterfly::radix4_narrow_f32_avx_version,
                    8 => butterfly::radix8_narrow_f32_avx_version,
                    radix => panic!("invalid radix: {}", radix),
                };
                return Self {
                    parameters,
                    func,
                    twiddle_offset,
                };
            };
        }

        // SSE narrow implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if safe_simd::x86::sse::Sse::new().is_some() {
                let func: StepFn<f32> = match parameters.radix {
                    2 => butterfly::radix2_narrow_f32_sse3_version,
                    3 => butterfly::radix3_narrow_f32_sse3_version,
                    4 => butterfly::radix4_narrow_f32_sse3_version,
                    8 => butterfly::radix8_narrow_f32_sse3_version,
                    radix => panic!("invalid radix: {}", radix),
                };
                return Self {
                    parameters,
                    func,
                    twiddle_offset,
                };
            };
        }

        // Generic implementations
        let func: StepFn<f32> = match parameters.radix {
            2 => butterfly::radix2_narrow_f32_default_version,
            3 => butterfly::radix3_narrow_f32_default_version,
            4 => butterfly::radix4_narrow_f32_default_version,
            8 => butterfly::radix8_narrow_f32_default_version,
            radix => panic!("invalid radix: {}", radix),
        };
        Self {
            parameters,
            func,
            twiddle_offset,
        }
    }
}

/// Implements a mixed-radix Stockham autosort algorithm for multiples of 2 and 3.
pub struct Autosort<T, Steps, Twiddles, Work> {
    size: usize,
    steps: Steps,
    twiddles: (Twiddles, Twiddles),
    work: RefCell<Work>,
    real_type: PhantomData<T>,
}

impl<T, Steps, Twiddles, Work> Autosort<T, Steps, Twiddles, Work>
where
    T: FftFloat,
    Twiddles: AsMut<[Complex<T>]>,
    Steps: AsMut<[Step<T>]>,
{
    pub fn new_from_parameters<F>(
        parameters: &[StepParameters],
        mut steps: Steps,
        mut twiddles: (Twiddles, Twiddles),
        work: Work,
        step_init: F,
    ) -> Self
    where
        F: Fn(StepParameters, usize) -> Step<T>,
    {
        // Initialize twiddles and steps
        let mut twiddle_offset = 0;
        for (index, step) in parameters.iter().enumerate() {
            // initialize twiddles
            step.initialize_twiddles(
                &mut twiddles.0.as_mut()[twiddle_offset..],
                &mut twiddles.1.as_mut()[twiddle_offset..],
            );

            // initialize step
            steps.as_mut()[index] = step_init(*step, twiddle_offset);

            // advance twiddles
            twiddle_offset += step.size;
        }
        Self {
            size: parameters[0].size,
            steps,
            twiddles,
            work: RefCell::new(work),
            real_type: PhantomData,
        }
    }
}

/*

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

*/
