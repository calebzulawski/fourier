//! Implementation of a mixed-radix Stockham autosort FFT.

mod avx_optimization;
mod butterfly;

use crate::array::Array;
use crate::autosort::butterfly::{apply_butterfly, Butterfly2, Butterfly3, Butterfly4, Butterfly8};
use crate::fft::{Fft, Transform};
use crate::float::Float;
use crate::twiddle::compute_twiddle;
use arch_types::Features;
use core::cell::RefCell;
use core::marker::PhantomData;
use generic_simd::vector::{width, Complex, Handle, SizedHandle, Vector};
use num_complex as nc;
use num_traits::One as _;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{boxed::Box, vec::Vec};

/// Represents the parameters of a single FFT step
#[derive(Copy, Clone, Default, Debug)]
pub struct Step {
    pub size: usize,
    pub stride: usize,
    pub count: usize,
    pub twiddle_offset: usize,
}

impl Step {
    fn initialize_twiddles<T: Float>(
        &self,
        radix: usize,
        forward: &mut [nc::Complex<T>],
        inverse: &mut [nc::Complex<T>],
    ) {
        let forward = &mut forward[self.twiddle_offset..];
        let inverse = &mut inverse[self.twiddle_offset..];
        let m = self.size / radix;
        for i in 0..m {
            forward[i * radix] = nc::Complex::one();
            inverse[i * radix] = nc::Complex::one();
            for j in 1..radix {
                forward[i * radix + j] = compute_twiddle(i * j, self.size, true);
                inverse[i * radix + j] = compute_twiddle(i * j, self.size, false);
            }
        }
    }
}

const NUM_STEPS: usize = 5;
const RADICES: [usize; NUM_STEPS] = [4, 8, 4, 3, 2];
type Steps = [Step; NUM_STEPS];

/// Determines the steps for a particular FFT size.
///
/// Returns the steps and the total number of twiddles.
pub fn steps(size: usize) -> Option<(Steps, usize)> {
    let mut current_size = size;
    let mut current_stride = 1;
    let mut current_twiddle_offset = 0;

    let mut steps = Steps::default();

    // First step is radix 4 (helps performance)
    if current_size % 4 == 0 {
        steps[0] = Step {
            size: current_size,
            stride: current_stride,
            count: 1,
            twiddle_offset: current_twiddle_offset,
        };
        current_size /= 4;
        current_stride *= 4;
        current_twiddle_offset += current_size;
    }

    for (index, radix) in RADICES.iter().copied().enumerate().skip(1) {
        let size = current_size;
        let stride = current_stride;
        let twiddle_offset = current_twiddle_offset;
        let mut count = 0;
        while current_size % radix == 0 {
            count += 1;
            current_size /= radix;
            current_stride *= radix;
            current_twiddle_offset += current_size;
        }
        steps[index] = Step {
            size,
            stride,
            count,
            twiddle_offset,
        };
    }
    if current_size == 1 {
        Some((steps, current_twiddle_offset))
    } else {
        None
    }
}

/// Implements a mixed-radix Stockham autosort algorithm for multiples of 2 and 3.
pub struct Autosort<T, Twiddles, Work> {
    size: usize,
    steps: Steps,
    twiddles: (Twiddles, Twiddles),
    work: RefCell<Work>,
    real_type: PhantomData<T>,
}

impl<T, Twiddles, Work> core::fmt::Debug for Autosort<T, Twiddles, Work> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("Autosort")
            .field("size", &self.size)
            .field("steps", &self.steps)
            .finish()
    }
}

impl<T, Twiddles, Work> Autosort<T, Twiddles, Work>
where
    T: Float,
    Twiddles: Array<nc::Complex<T>>,
    Work: Array<nc::Complex<T>>,
{
    /// Constructs an FFT.
    pub fn new(size: usize) -> Option<Self> {
        if let Some((steps, num_twiddles)) = steps(size) {
            let mut forward_twiddles = Twiddles::new(num_twiddles);
            let mut inverse_twiddles = Twiddles::new(num_twiddles);
            let work = RefCell::new(Work::new(size));

            // Initialize twiddles and steps
            for (radix, step) in RADICES.iter().copied().zip(steps.iter()) {
                // initialize twiddles
                step.initialize_twiddles(
                    radix,
                    &mut forward_twiddles.as_mut()[step.twiddle_offset..],
                    &mut inverse_twiddles.as_mut()[step.twiddle_offset..],
                );
            }
            Some(Self {
                size,
                steps,
                twiddles: (forward_twiddles, inverse_twiddles),
                work,
                real_type: PhantomData,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    fn impl_in_place<H>(&self, handle: H, input: &mut [nc::Complex<T>], transform: Transform)
    where
        H: Handle<nc::Complex<T>>,
        <H as SizedHandle<nc::Complex<T>, width::W1>>::Vector: Complex,
        <H as SizedHandle<nc::Complex<T>, width::W2>>::Vector: Complex,
        <H as SizedHandle<nc::Complex<T>, width::W4>>::Vector: Complex,
        <H as SizedHandle<nc::Complex<T>, width::W8>>::Vector: Complex,
    {
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
        let mut current_size = self.size;
        let mut current_stride = 1;
        let mut current_twiddle_offset = 0;
        for (step, radix) in self.steps.iter().zip(&RADICES) {
            for _ in 0..step.count {
                // determine input and output
                let (from, to): (&mut _, &mut _) = if data_in_work {
                    (work, input)
                } else {
                    (input, work)
                };

                // apply step
                match radix {
                    2 => apply_butterfly(
                        Butterfly2,
                        handle,
                        from,
                        to,
                        current_size,
                        current_stride,
                        &twiddles.as_ref()[current_twiddle_offset..],
                        transform.is_forward(),
                    ),
                    3 => apply_butterfly(
                        Butterfly3,
                        handle,
                        from,
                        to,
                        current_size,
                        current_stride,
                        &twiddles.as_ref()[current_twiddle_offset..],
                        transform.is_forward(),
                    ),
                    4 => apply_butterfly(
                        Butterfly4,
                        handle,
                        from,
                        to,
                        current_size,
                        current_stride,
                        &twiddles.as_ref()[current_twiddle_offset..],
                        transform.is_forward(),
                    ),
                    8 => apply_butterfly(
                        Butterfly8,
                        handle,
                        from,
                        to,
                        current_size,
                        current_stride,
                        &twiddles.as_ref()[current_twiddle_offset..],
                        transform.is_forward(),
                    ),
                    _ => panic!("unsupported radix"),
                }

                // update state
                current_twiddle_offset += current_size;
                current_size /= radix;
                current_stride *= radix;

                // swap buffers
                data_in_work = !data_in_work;
            }
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

macro_rules! implement {
    {
        $handle:ident, $type:ty
    } => {
        impl<Twiddles, Work> Autosort<$type, Twiddles, Work>
        where
            Twiddles: Array<nc::Complex<$type>>,
            Work: Array<nc::Complex<$type>>,
        {
            #[generic_simd::dispatch($handle)]
            fn impl_in_place_dispatch(&self, input: &mut [nc::Complex<$type>], transform: Transform) {
                self.impl_in_place($handle, input, transform);
            }
        }

        impl<Twiddles, Work> Fft for Autosort<$type, Twiddles, Work>
        where
            Twiddles: Array<nc::Complex<$type>>,
            Work: Array<nc::Complex<$type>>,
        {
            type Real = $type;

            fn size(&self) -> usize {
                self.size
            }

            fn transform_in_place(&self, input: &mut [nc::Complex<$type>], transform: Transform) {
                self.impl_in_place_dispatch(input, transform);
            }
        }
    }
}
implement! { handle, f32 }
implement! { handle, f64 }

/// Implementation of the Stockham autosort algorithm backed by heap allocations.
///
/// Requires the `std` or `alloc` features.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type HeapAutosort<T> = Autosort<T, Box<[nc::Complex<T>]>, Box<[nc::Complex<T>]>>;
