//! Implementation of a mixed-radix Stockham autosort FFT.

mod avx_optimization;
mod butterfly;

use crate::array::Array;
use crate::fft::{Fft, Transform};
use crate::float::Float;
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
    fn initialize_twiddles<T: Float>(
        &self,
        forward: &mut [Complex<T>],
        inverse: &mut [Complex<T>],
    ) {
        let m = self.size / self.radix;
        for i in 0..m {
            forward[i * self.radix] = Complex::one();
            inverse[i * self.radix] = Complex::one();
            for j in 1..self.radix {
                forward[i * self.radix + j] = compute_twiddle(i * j, self.size, true);
                inverse[i * self.radix + j] = compute_twiddle(i * j, self.size, false);
            }
        }
    }
}

/// Determines the steps for a particular FFT size.
pub fn steps<E: Default + Extend<StepParameters>>(size: usize) -> Option<E> {
    let mut steps = E::default();
    let mut current_size = size;
    let mut stride = 1;

    // First step is radix 4 (helps performance)
    if current_size % 4 == 0 {
        steps.extend(core::iter::once(StepParameters {
            size: current_size,
            radix: 4,
            stride,
        }));
        current_size /= 4;
        stride *= 4;
    }

    for radix in [8, 4, 3, 2].iter().copied() {
        while current_size % radix == 0 {
            steps.extend(core::iter::once(StepParameters {
                size: current_size,
                radix,
                stride,
            }));
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
#[derive(Clone)]
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

impl<T> core::fmt::Debug for Step<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        write!(
            f,
            "(radix: {}, size: {}, stride: {})",
            self.parameters.radix, self.parameters.size, self.parameters.stride
        )
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
        let twiddles = &twiddles[self.twiddle_offset..(self.twiddle_offset + self.parameters.size)];
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

/// Constructs a step from parameters and a twiddle offset.
///
/// This trait exists to specialize construction of `Step`
pub trait StepInit {
    fn new(parameters: StepParameters, twiddle_offset: usize) -> Self;
}

impl StepInit for Step<f32> {
    fn new(parameters: StepParameters, twiddle_offset: usize) -> Self {
        assert!(parameters.size % parameters.radix == 0);

        // Optimization
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if parameters.stride == 1 && parameters.radix == 4 {
                return Self {
                    parameters,
                    func: crate::autosort::avx_optimization::radix_4_stride_1_avx_f32,
                    twiddle_offset,
                };
            }
        }

        // AVX wide implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if parameters.stride >= safe_simd::x86::avx::Vcf32::width()
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
            if parameters.stride >= safe_simd::x86::sse::Vcf32::width()
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

impl StepInit for Step<f64> {
    fn new(parameters: StepParameters, twiddle_offset: usize) -> Self {
        assert!(parameters.size % parameters.radix == 0);

        // AVX wide implementations
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if parameters.stride >= safe_simd::x86::avx::Vcf64::width()
                && safe_simd::x86::avx::Avx::new().is_some()
            {
                let func: StepFn<f64> = match parameters.radix {
                    2 => butterfly::radix2_wide_f64_avx_version,
                    3 => butterfly::radix3_wide_f64_avx_version,
                    4 => butterfly::radix4_wide_f64_avx_version,
                    8 => butterfly::radix8_wide_f64_avx_version,
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
            if parameters.stride >= safe_simd::x86::sse::Vcf64::width()
                && safe_simd::x86::sse::Sse::new().is_some()
            {
                let func: StepFn<f64> = match parameters.radix {
                    2 => butterfly::radix2_wide_f64_sse3_version,
                    3 => butterfly::radix3_wide_f64_sse3_version,
                    4 => butterfly::radix4_wide_f64_sse3_version,
                    8 => butterfly::radix8_wide_f64_sse3_version,
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
                let func: StepFn<f64> = match parameters.radix {
                    2 => butterfly::radix2_narrow_f64_avx_version,
                    3 => butterfly::radix3_narrow_f64_avx_version,
                    4 => butterfly::radix4_narrow_f64_avx_version,
                    8 => butterfly::radix8_narrow_f64_avx_version,
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
                let func: StepFn<f64> = match parameters.radix {
                    2 => butterfly::radix2_narrow_f64_sse3_version,
                    3 => butterfly::radix3_narrow_f64_sse3_version,
                    4 => butterfly::radix4_narrow_f64_sse3_version,
                    8 => butterfly::radix8_narrow_f64_sse3_version,
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
        let func: StepFn<f64> = match parameters.radix {
            2 => butterfly::radix2_narrow_f64_default_version,
            3 => butterfly::radix3_narrow_f64_default_version,
            4 => butterfly::radix4_narrow_f64_default_version,
            8 => butterfly::radix8_narrow_f64_default_version,
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

impl<T, Steps, Twiddles, Work> core::fmt::Debug for Autosort<T, Steps, Twiddles, Work>
where
    Steps: AsRef<[Step<T>]>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("Autosort")
            .field("size", &self.size)
            .field("steps", &self.steps.as_ref())
            .finish()
    }
}

impl<T, Steps, Twiddles, Work> Autosort<T, Steps, Twiddles, Work>
where
    T: Float,
    Twiddles: AsMut<[Complex<T>]>,
    Steps: AsMut<[Step<T>]>,
    Work: AsMut<[Complex<T>]>,
    Step<T>: StepInit,
{
    /// Constructs an FFT from parameters.
    ///
    /// * `steps` must be the same size as `parameters`
    /// * `twiddles` must be large enough to store all twiddles (as determined by `num_twiddles`)
    /// * `work` must be the same size as the FFT size
    pub fn new_from_parameters(
        parameters: &[StepParameters],
        mut steps: Steps,
        mut twiddles: (Twiddles, Twiddles),
        mut work: Work,
    ) -> Self {
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
            steps.as_mut()[index] = Step::new(*step, twiddle_offset);

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

impl<T, Steps, Twiddles, Work> Autosort<T, Steps, Twiddles, Work>
where
    T: Float,
    Twiddles: Array<Complex<T>>,
    Steps: Array<Step<T>>,
    Work: Array<Complex<T>>,
    Step<T>: StepInit,
{
    /// Constructs an FFT over types that are `Extend`.
    pub fn new_with_extend<E: Default + Extend<StepParameters> + AsRef<[StepParameters]>>(
        size: usize,
    ) -> Option<Self> {
        if let Some(step_parameters) = steps::<E>(size) {
            let num_twiddles = num_twiddles(step_parameters.as_ref());
            let forward_twiddles = Twiddles::new(num_twiddles);
            let inverse_twiddles = Twiddles::new(num_twiddles);
            let steps = Steps::new(step_parameters.as_ref().len());
            let work = Work::new(size);
            Some(Self::new_from_parameters(
                step_parameters.as_ref(),
                steps,
                (forward_twiddles, inverse_twiddles),
                work,
            ))
        } else {
            None
        }
    }
}

/// Implementation of the Stockham autosort algorithm backed by heap allocations.
///
/// Requires the `std` or `alloc` features.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type HeapAutosort<T> = Autosort<T, Box<[Step<T>]>, Box<[Complex<T>]>, Box<[Complex<T>]>>;

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Float> HeapAutosort<T>
where
    Step<T>: StepInit,
{
    /// Constructs a Stockham autosort FFT with the specified size.
    pub fn new(size: usize) -> Option<Self> {
        Self::new_with_extend::<Vec<_>>(size)
    }
}

impl<T, Steps, Twiddles, Work> Fft for Autosort<T, Steps, Twiddles, Work>
where
    T: Float,
    Twiddles: AsRef<[Complex<T>]>,
    Steps: AsRef<[Step<T>]>,
    Work: AsMut<[Complex<T>]>,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<T>], transform: Transform) {
        assert_eq!(input.len(), self.size);

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
