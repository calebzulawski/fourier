use crate::autosort::StepParameters;
use crate::fft::{Fft, Transform};
use core::cell::RefCell;
use num_complex::Complex;

#[derive(Debug)]
pub struct Power2<T> {
    twiddles: (Vec<num_complex::Complex<T>>, Vec<num_complex::Complex<T>>),
    work: RefCell<Vec<num_complex::Complex<T>>>,
}

impl Power2<f32> {
    pub fn new() -> Self {
        let mut forward = vec![num_complex::Complex::<f32>::default(); 256 + 64 + 8];
        let mut inverse = vec![num_complex::Complex::<f32>::default(); 256 + 64 + 8];

        StepParameters {
            size: 256,
            stride: 1,
            radix: 4,
        }
        .initialize_twiddles(&mut forward, &mut inverse);

        StepParameters {
            size: 64,
            stride: 4,
            radix: 8,
        }
        .initialize_twiddles(&mut forward[256..], &mut inverse[256..]);

        StepParameters {
            size: 8,
            stride: 32,
            radix: 8,
        }
        .initialize_twiddles(&mut forward[320..], &mut inverse[320..]);

        Self {
            twiddles: (forward, inverse),
            work: RefCell::new(vec![num_complex::Complex::<f32>::default(); 256]),
        }
    }

    #[multiversion::target("[x86|x86_64]+avx")]
    #[safe_inner]
    unsafe fn fft(&self, input: &mut [Complex<f32>], transform: Transform) {
        assert_eq!(input.len(), 256);

        // Obtain the work buffer
        let mut work_ref = self.work.borrow_mut();
        let work = work_ref.as_mut();

        // Select the twiddles for this operation
        let twiddles = if transform.is_forward() {
            &self.twiddles.0
        } else {
            &self.twiddles.1
        };

        unsafe {
            crate::autosort::avx_optimization::radix_4_stride_1_avx_f32(
                input,
                work,
                256,
                1,
                &twiddles[..256],
                true,
            );
            crate::autosort::butterfly::radix8_wide_f32_avx_version(
                work,
                input,
                64,
                4,
                &twiddles[256..320],
                true,
            );
            crate::autosort::butterfly::radix8_wide_f32_avx_version(
                input,
                work,
                8,
                32,
                &twiddles[320..],
                true,
            );
        }

        // Finish operation by scaling and moving data if necessary
        if let Some(scale) = match transform {
            Transform::Fft | Transform::UnscaledIfft => None,
            Transform::Ifft => Some(1f32 / 256f32),
            Transform::SqrtScaledFft | Transform::SqrtScaledIfft => Some(1f32 / 16f32),
        } {
            for x in input.iter_mut() {
                *x *= scale;
            }
        }
    }
}

impl Fft for Power2<f32> {
    type Real = f32;

    fn size(&self) -> usize {
        256
    }

    fn transform_in_place(&self, input: &mut [num_complex::Complex<f32>], transform: Transform) {
        unsafe {
            self.fft(input, transform);
        }
    }
}
