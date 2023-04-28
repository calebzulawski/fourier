mod butterfly;

use crate::{scalar::Scalar, Fft, Transform};
use core::{
    cell::RefCell,
    simd::{LaneCount, Simd, SupportedLaneCount},
};
use num_complex as nc;
use num_traits::{Float, FromPrimitive, One};
use simd_complex::SimdComplex;
use simd_traits::{num::Signed, swizzle::Shuffle, Vector};

const NUM_RADICES: usize = 5;
const RADICES: [usize; NUM_RADICES] = [4, 8, 4, 3, 2];

#[inline]
fn compute_twiddle<T, const FORWARD: bool>(index: usize, size: usize) -> num_complex::Complex<T>
where
    T: Copy + Float + FromPrimitive,
{
    let theta = (index * 2) as f64 * core::f64::consts::PI / size as f64;
    let twiddle = num_complex::Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    );
    if FORWARD {
        twiddle
    } else {
        twiddle.conj()
    }
}

fn initialize_twiddles<T: Scalar>(
    mut size: usize,
    counts: [usize; NUM_RADICES],
) -> (Vec<nc::Complex<T>>, Vec<nc::Complex<T>>) {
    let (mut forward_twiddles, mut inverse_twiddles) = (Vec::new(), Vec::new());
    for (radix, count) in RADICES.iter().zip(&counts) {
        for _ in 0..*count {
            let m = size / radix;
            for i in 0..m {
                forward_twiddles.push(nc::Complex::<T>::one());
                inverse_twiddles.push(nc::Complex::<T>::one());
                for j in 1..*radix {
                    forward_twiddles.push(compute_twiddle::<T, true>(i * j, size));
                    inverse_twiddles.push(compute_twiddle::<T, false>(i * j, size));
                }
            }
            size /= radix;
        }
    }
    (forward_twiddles, inverse_twiddles)
}

/// Implements a mixed-radix Stockham autosort algorithm for multiples of 2 and 3.
pub struct Autosort<T> {
    size: usize,
    counts: [usize; NUM_RADICES],
    forward_twiddles: Vec<nc::Complex<T>>,
    inverse_twiddles: Vec<nc::Complex<T>>,
    work: RefCell<Vec<nc::Complex<T>>>,
}

impl<T: Scalar> Autosort<T> {
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
            let (forward_twiddles, inverse_twiddles) = initialize_twiddles(size, counts);
            let work = vec![Default::default(); size];
            Some(Self {
                size,
                counts,
                forward_twiddles,
                inverse_twiddles,
                work: RefCell::new(work),
            })
        } else {
            None
        }
    }
}

impl<T> Fft for Autosort<T>
where
    T: Scalar,
    T::Mask: PartialEq,
    Simd<T, 1>: Vector<Scalar = T> + Signed,
    Simd<T, 4>: Vector<Scalar = T> + Signed,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [nc::Complex<T>], transform: Transform) {
        let mut work = self.work.borrow_mut();
        apply_steps(&self, input, work.as_mut(), transform);
    }
}

/// Call one step of the FFT with the given size and stride.
///
/// # Safety
/// The input and output lengths must match the size * stride.
/// The twiddles must match the size.
#[inline(always)]
unsafe fn step<T, const LANES: usize, const RADIX: usize, const FORWARD: bool>(
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    twiddles: &[nc::Complex<T>],
    size: usize,
    stride: usize,
) where
    T: Scalar,
    Simd<T, LANES>: Vector<Scalar = T> + Signed + Shuffle,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert!(core::mem::size_of::<Simd<T, LANES>>() == core::mem::size_of::<[T; LANES]>());
    debug_assert!(input.len() == size * stride);
    debug_assert!(output.len() == size * stride);
    debug_assert!(twiddles.len() >= size);
    debug_assert!(stride >= LANES);

    // TODO AVX optimization

    let m = size / RADIX;

    for i in 0..m {
        let twiddles = {
            let mut step_twiddles = [SimdComplex::<T, LANES>::default(); RADIX];
            for k in 1..RADIX {
                let index = i * RADIX + k;
                // TODO: prove safety
                // debug_assert!(index < twiddles.len())
                // let twiddle =
                //     unsafe {
                //     twiddles.as_ptr().add(index).read())
                // };
                let twiddle = twiddles[index];
                step_twiddles[k] = SimdComplex::splat(twiddle);
            }
            step_twiddles
        };

        let mut step = |j: usize| {
            // Load full vectors
            let mut scratch = [SimdComplex::<T, LANES>::default(); RADIX];
            let load = unsafe { input.as_ptr().add(j + stride * i) };
            for k in 0..RADIX {
                debug_assert!(j + stride * (i + k * m) + LANES <= input.len());
                let packed = unsafe {
                    load.add(stride * k * m)
                        .cast::<[Simd<T, LANES>; 2]>()
                        .read_unaligned()
                };
                scratch[k] = SimdComplex::from_packed(packed);
            }

            // Butterfly with optional twiddles
            scratch = butterfly::butterfly::<T, Simd<T, LANES>, RADIX, FORWARD>(scratch);
            if size != RADIX {
                for k in 1..RADIX {
                    scratch[k] = scratch[k] * twiddles[k];
                }
            }

            // Store full vectors
            let store = unsafe { output.as_mut_ptr().add(j + RADIX * stride * i) };
            for k in 0..RADIX {
                debug_assert!(j + stride * (i * RADIX + k) + LANES <= output.len());
                let packed = scratch[k].to_packed();
                unsafe {
                    (store.add(stride * k).cast::<[Simd<T, LANES>; 2]>()).write_unaligned(packed)
                };
            }
        };

        if LANES > 1 {
            let full_count = ((stride - 1) / LANES) * LANES;
            let final_offset = stride - LANES;
            for j in (0..full_count)
                .step_by(LANES)
                .chain(core::iter::once(final_offset))
            {
                step(j)
            }
        } else {
            for j in 0..stride {
                step(j)
            }
        }
    }
}

fn apply_steps<T>(
    autosort: &Autosort<T>,
    input: &mut [nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    transform: Transform,
) where
    T: Scalar,
    T::Mask: PartialEq,
    Simd<T, 1>: Vector<Scalar = T> + Signed,
    Simd<T, 4>: Vector<Scalar = T> + Signed,
{
    assert_eq!(input.len(), autosort.size);
    assert_eq!(output.len(), autosort.size);

    let mut twiddles = if transform.is_forward() {
        autosort.forward_twiddles.as_ref()
    } else {
        autosort.inverse_twiddles.as_ref()
    };

    const LANES: usize = 4; // FIXME use the preferred width for the target

    fn pick_step<T, const LANES: usize>(
        from: &mut [nc::Complex<T>],
        to: &mut [nc::Complex<T>],
        forward: bool,
        radix: usize,
        size: usize,
        stride: usize,
        twiddles: &[nc::Complex<T>],
    ) where
        T: Scalar,
        T::Mask: PartialEq,
        Simd<T, LANES>: Vector<Scalar = T> + Signed,
        LaneCount<LANES>: SupportedLaneCount,
    {
        unsafe {
            if forward {
                match radix {
                    8 => step::<T, LANES, 8, true>(from, to, twiddles, size, stride),
                    4 => step::<T, LANES, 4, true>(from, to, twiddles, size, stride),
                    3 => step::<T, LANES, 3, true>(from, to, twiddles, size, stride),
                    2 => step::<T, LANES, 2, true>(from, to, twiddles, size, stride),
                    _ => unimplemented!("unsupported radix"),
                }
            } else {
                match radix {
                    8 => step::<T, LANES, 8, false>(from, to, twiddles, size, stride),
                    4 => step::<T, LANES, 4, false>(from, to, twiddles, size, stride),
                    3 => step::<T, LANES, 3, false>(from, to, twiddles, size, stride),
                    2 => step::<T, LANES, 2, false>(from, to, twiddles, size, stride),
                    _ => unimplemented!("unsupported radix"),
                }
            }
        }
    }

    let mut size = autosort.size;
    let mut stride = 1;
    let mut data_in_output = false;
    for (radix, iterations) in RADICES.iter().zip(autosort.counts) {
        let mut iteration = 0;

        // Use partial loads until the stride is large enough
        while stride < LANES && iteration < iterations {
            let (from, to): (&mut _, &mut _) = if data_in_output {
                (output, input)
            } else {
                (input, output)
            };
            pick_step::<T, 1>(
                from,
                to,
                transform.is_forward(),
                *radix,
                size,
                stride,
                &twiddles,
            );
            size /= radix;
            stride *= radix;
            twiddles = &twiddles[size * radix..];
            iteration += 1;
            data_in_output = !data_in_output;
        }

        for _ in iteration..iterations {
            let (from, to): (&mut _, &mut _) = if data_in_output {
                (output, input)
            } else {
                (input, output)
            };
            pick_step::<T, LANES>(
                from,
                to,
                transform.is_forward(),
                *radix,
                size,
                stride,
                &twiddles,
            );
            size /= radix;
            stride *= radix;
            twiddles = &twiddles[size * radix..];
            data_in_output = !data_in_output;
        }
    }
    if let Some(scale) = match transform {
        Transform::Fft | Transform::UnscaledIfft => None,
        Transform::Ifft => Some(T::one() / T::from_usize(input.len()).unwrap()),
        Transform::SqrtScaledFft | Transform::SqrtScaledIfft => {
            Some(T::one() / T::from_usize(input.len()).unwrap().sqrt())
        }
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
