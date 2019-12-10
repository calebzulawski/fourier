use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) mod pow2;
pub(crate) mod prime_factor;
mod radix2;
mod radix3;
mod radix4;
mod radix8;
use crate::vector::ComplexVector;

#[inline(always)]
fn zeroed_array<T, Vector: crate::vector::ComplexVector<Float = T>, const SIZE: usize>(
) -> [Vector; SIZE] {
    // MaybeUninit is a workaround for not being able to init generic arrays
    let mut array: [MaybeUninit<Vector>; SIZE] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..SIZE {
        array[i] = MaybeUninit::new(unsafe { Vector::zero() });
    }
    unsafe { (&array as *const _ as *const [Vector; SIZE]).read() }
}

trait Butterfly<T, const RADIX: usize>: Sized {
    fn new(forward: bool) -> Self;

    unsafe fn apply<Vector: ComplexVector<Float = T>>(
        &self,
        input: [Vector; RADIX],
    ) -> [Vector; RADIX];

    #[inline(always)]
    unsafe fn apply_step_full<Vector: ComplexVector<Float = T>>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        size: usize,
        stride: usize,
        remaining_twiddles: &[Complex<T>],
    ) {
        fn safe<T, Bfly, Vector, const RADIX: usize>(
            bfly: &Bfly,
            input: &[Complex<T>],
            output: &mut [Complex<T>],
            size: usize,
            stride: usize,
            remaining_twiddles: &[Complex<T>],
        ) where
            Bfly: Butterfly<T, { RADIX }>,
            Vector: ComplexVector<Float = T>,
        {
            let m = size / RADIX;

            let full_count = ((stride - 1) / Vector::WIDTH) * Vector::WIDTH;
            let final_offset = stride - Vector::WIDTH;

            for i in 0..m {
                // Load twiddle factors
                let twiddles = {
                    let mut twiddles = zeroed_array::<T, Vector, { RADIX }>();
                    for k in 1..RADIX {
                        let twiddle = &unsafe { remaining_twiddles.get_unchecked(i + (k - 1) * m) };
                        twiddles[k] = unsafe { Vector::broadcast(&twiddle) };
                    }
                    twiddles
                };

                // Loop over full vectors, with a final overlapping vector
                for j in (0..full_count)
                    .step_by(Vector::WIDTH)
                    .chain(std::iter::once(final_offset))
                {
                    // Load full vectors
                    let mut scratch = zeroed_array::<T, Vector, { RADIX }>();
                    let load = unsafe { input.as_ptr().add(j + stride * i) };
                    for k in 0..RADIX {
                        scratch[k] = unsafe { Vector::load(load.add(stride * k * m)) };
                    }

                    // Butterfly with optional twiddles
                    scratch = unsafe { bfly.apply(scratch) };
                    if size != RADIX {
                        for k in 1..RADIX {
                            scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                        }
                    }

                    // Store full vectors
                    let store = unsafe { output.as_mut_ptr().add(j + RADIX * stride * i) };
                    for k in 0..RADIX {
                        unsafe { scratch[k].store(store.add(stride * k)) };
                    }
                }
            }
        }
        safe::<T, Self, Vector, { RADIX }>(self, input, output, size, stride, remaining_twiddles)
    }

    #[inline(always)]
    unsafe fn apply_step_partial<Vector: ComplexVector<Float = T>>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        size: usize,
        stride: usize,
        remaining_twiddles: &[Complex<T>],
    ) {
        fn safe<T, Bfly, Vector, const RADIX: usize>(
            bfly: &Bfly,
            input: &[Complex<T>],
            output: &mut [Complex<T>],
            size: usize,
            stride: usize,
            remaining_twiddles: &[Complex<T>],
        ) where
            Bfly: Butterfly<T, { RADIX }>,
            Vector: ComplexVector<Float = T>,
        {
            let m = size / RADIX;

            for i in 0..m {
                // Load twiddle factors
                let twiddles = {
                    let mut twiddles = zeroed_array::<T, Vector, { RADIX }>();
                    for k in 1..RADIX {
                        let twiddle = &unsafe { remaining_twiddles.get_unchecked(i + (k - 1) * m) };
                        twiddles[k] = unsafe { Vector::broadcast(&twiddle) };
                    }
                    twiddles
                };

                let load = unsafe { input.as_ptr().add(stride * i) };
                let store = unsafe { output.as_mut_ptr().add(RADIX * stride * i) };
                for j in 0..stride {
                    // Load a partial vector
                    let mut scratch = zeroed_array::<T, Vector, { RADIX }>();
                    for k in 0..RADIX {
                        scratch[k] = unsafe { Vector::load1(load.add(stride * k * m + j)) };
                    }

                    // Butterfly with optional twiddles
                    scratch = unsafe { bfly.apply(scratch) };
                    if size != RADIX {
                        for k in 1..RADIX {
                            scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                        }
                    }

                    // Store a partial vector
                    for k in 0..RADIX {
                        unsafe { scratch[k].store1(store.add(stride * k + j)) };
                    }
                }
            }
        }
        safe::<T, Self, Vector, { RADIX }>(self, input, output, size, stride, remaining_twiddles)
    }
}
