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

/// Represents the parameters of a single FFT step
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

/// Determines the steps for a particular FFT size.
///
/// Requires the `std` or `alloc` features.
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

/// Returns the number of twiddle factors for a particular FFT.
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

/// An FFT step.
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
    Work: AsMut<[Step<T>]>,
{
    /// Constructs an FFT from parameters.
    ///
    /// * `steps` must be the same size as `parameters
    /// * `twiddles` must be large enough to store all twiddles (as determined by `num_twiddles`)
    /// * `work` must be the same size as the FFT size
    pub fn new_from_parameters<F>(
        parameters: &[StepParameters],
        mut steps: Steps,
        mut twiddles: (Twiddles, Twiddles),
        mut work: Work,
        step_init: F,
    ) -> Self
    where
        F: Fn(StepParameters, usize) -> Step<T>,
    {
        let size = parameters[0].size;
        assert!(work.as_mut().len() == size);

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
            size,
            steps,
            twiddles,
            work: RefCell::new(work),
            real_type: PhantomData,
        }
    }
}

impl<T, Steps, Twiddles, Work> Fft for Autosort<T, Steps, Twiddles, Work>
where
    T: FftFloat,
    Twiddles: AsRef<[Complex<T>]>,
    Steps: AsRef<[Step<T>]>,
    Work: AsMut<[Complex<T>]>,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<T>], transform: Transform) {
        // Obtain the work buffer
        let mut work_ref = self.work.borrow_mut();
        let work = work_ref.as_mut();

        // Select the twiddles for this operation
        let twiddles = if transform.is_forward() {
            &self.twiddles.0
        } else {
            &self.twiddles.1
        };

        // Apply steps with data ping-ponging between work and input buffer
        let mut data_in_work = false;
        for step in self.steps.as_ref() {
            // determine input and output
            let (from, to): (&mut _, &mut _) = if data_in_work {
                (work, input)
            } else {
                (input, work)
            };

            // apply step
            step.apply(from, to, twiddles.as_ref(), transform.is_forward());

            // swap buffers
            data_in_work = !data_in_work;
        }

        // Finish operation by scaling and moving data if necessary
        if let Some(scale) = match transform {
            Transform::Fft | Transform::UnscaledIfft => None,
            Transform::Ifft => Some(T::one() / T::from_usize(self.size).unwrap()),
            Transform::SqrtScaledFft | Transform::SqrtScaledIfft => {
                Some(T::one() / T::from_usize(self.size).unwrap().sqrt())
            }
        } {
            if data_in_work {
                for (x, y) in work.iter().zip(input.iter_mut()) {
                    *y = x * scale;
                }
            } else {
                for x in input.iter_mut() {
                    *x *= scale;
                }
            }
        } else {
            if data_in_work {
                input.copy_from_slice(work);
            }
        }
    }
}
