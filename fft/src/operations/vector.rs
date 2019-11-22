use num_complex::Complex;
use std::mem::MaybeUninit;

trait Butterfly<T, const RADIX: usize> {
    fn apply(&self, input: [T; RADIX], forward: bool) -> [T; RADIX];
}

struct ButterflyStage<T, Bfly, const RADIX: usize> {
    butterfly: Bfly,
    twiddles: Vec<Complex<T>>,
    size: usize,
    stride: usize,
}

fn zeroed_array<T, Vector: crate::vector::ComplexVector<Float = T>, const RADIX: usize>(
) -> [Vector; RADIX] {
    // MaybeUninit is a workaround for not being able to init generic arrays
    let mut array: [MaybeUninit<Vector>; RADIX] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..RADIX {
        array[i] = MaybeUninit::new(unsafe { Vector::zero() });
    }
    unsafe { (&array as *const _ as *const [Vector; RADIX]).read() }
}

impl<T: Copy, Bfly, const RADIX: usize> ButterflyStage<T, Bfly, RADIX> {
    fn apply<Vector>(&self, input: &[Complex<T>], output: &mut [Complex<T>], forward: bool)
    where
        Vector: crate::vector::ComplexVector<Float = T>,
        Bfly: Butterfly<Vector, RADIX>,
    {
        assert_eq!(input.len(), self.size * self.stride);
        assert_eq!(output.len(), self.size * self.stride);

        let m = self.size / RADIX;
        let full_count = (self.stride / Vector::WIDTH) * Vector::WIDTH;
        let partial_count = self.stride - full_count;

        for i in 0..m {
            // Load twiddle factors
            let twiddles = {
                let mut twiddles = zeroed_array::<T, Vector, RADIX>();
                for k in 1..RADIX {
                    let twiddle = self.twiddles[i + k * m];
                    twiddles[k] = unsafe { Vector::broadcast(&twiddle) };
                }
                twiddles
            };

            // Loop over full vectors
            for j in (0..full_count).step_by(Vector::WIDTH) {
                // Load full vectors
                let mut scratch = zeroed_array::<T, Vector, RADIX>();
                let load = unsafe { input.as_ptr().add(j + self.stride * i) };
                for k in 0..RADIX {
                    scratch[k] = unsafe { Vector::load(load.add(self.stride * k * m)) };
                }

                // Butterfly with optional twiddles
                scratch = self.butterfly.apply(scratch, forward);
                if self.size != RADIX {
                    for k in 1..RADIX {
                        scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                    }
                }

                // Store full vectors
                let store = unsafe { output.as_mut_ptr().add(j + RADIX * self.stride * i) };
                for k in 0..RADIX {
                    unsafe { scratch[k].store(store.add(self.stride * k)) };
                }
            }

            // Apply the final partial vector
            if partial_count > 0 {
                // Load a partial vector
                let mut scratch = zeroed_array::<T, Vector, RADIX>();
                let load = unsafe { input.as_ptr().add(full_count + self.stride * i) };
                for k in 0..RADIX {
                    scratch[k] = unsafe {
                        Vector::partial_load(load.add(self.stride * k * m), partial_count)
                    };
                }

                // Butterfly with optional twiddles
                scratch = self.butterfly.apply(scratch, forward);
                if self.size != RADIX {
                    for k in 1..RADIX {
                        scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                    }
                }

                // Store a partial vector
                let store = unsafe {
                    output
                        .as_mut_ptr()
                        .add(full_count + RADIX * self.stride * i)
                };
                for k in 0..RADIX {
                    unsafe { scratch[k].partial_store(store.add(self.stride * k), partial_count) };
                }
            }
        }
    }
}
