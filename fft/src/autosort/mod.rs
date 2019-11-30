use crate::fft::Fft;
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use num_complex::Complex;
use std::mem::MaybeUninit;

mod radix2;
mod radix3;
mod radix4;
mod radix8;
use crate::vector::ComplexVector;
use radix2::Radix2;
use radix3::Radix3;
use radix4::Radix4;
use radix8::Radix8;

fn num_factors(factor: usize, mut value: usize) -> (usize, usize) {
    let mut count = 0;
    while value % factor == 0 {
        value /= factor;
        count += 1;
    }
    (count, value)
}

#[inline(always)]
fn zeroed_array<T, Vector: crate::vector::ComplexVector<Float = T>, const RADIX: usize>(
) -> [Vector; RADIX] {
    // MaybeUninit is a workaround for not being able to init generic arrays
    let mut array: [MaybeUninit<Vector>; RADIX] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..RADIX {
        array[i] = MaybeUninit::new(unsafe { Vector::zero() });
    }
    unsafe { (&array as *const _ as *const [Vector; RADIX]).read() }
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

struct ButterflyStage<T, Bfly, const RADIX: usize> {
    butterfly: Bfly,
    twiddles: Vec<Complex<T>>,
    size: usize,
    iterations: usize,
}

impl<T, Bfly, const RADIX: usize> ButterflyStage<T, Bfly, { RADIX }>
where
    T: FftFloat,
    Bfly: Butterfly<T, { RADIX }>,
{
    fn new(size: usize, iterations: usize, forward: bool) -> Self {
        let mut subsize = size;
        assert!(iterations != 0);
        let mut twiddles = Vec::new();
        for _ in 0..iterations {
            //assert_eq!(size % RADIX, 0);
            for i in 1..RADIX {
                let m = subsize / RADIX;
                for j in 0..m {
                    twiddles.push(compute_twiddle(i * j, subsize, forward));
                }
            }
            subsize /= RADIX;
        }
        Self {
            butterfly: Bfly::new(forward),
            twiddles,
            size,
            iterations,
        }
    }

    /*
    #[inline(always)]
    unsafe fn apply_out_of_place<Vector>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        work: &mut [Complex<T>],
    ) -> bool
    where
        Vector: crate::vector::ComplexVector<Float = T>,
    {
        #[inline(always)]
        fn safe<T, Bfly, Vector, const RADIX: usize>(
            bfly: &ButterflyStage<T, Bfly, { RADIX }>,
            input: &[Complex<T>],
            output: &mut [Complex<T>],
            work: &mut [Complex<T>],
        ) -> bool
        where
            T: FftFloat,
            Bfly: Butterfly<T, { RADIX }>,
            Vector: crate::vector::ComplexVector<Float = T>,
        {
            assert_eq!(input.len(), output.len());
            assert!(bfly.iterations > 0);
            let mut size = bfly.size;
            let mut all_twiddles: &[Complex<T>] = &bfly.twiddles;

            // First butterfly copies from the input buffer
            unsafe {
                bfly.butterfly
                    .apply_step::<Vector>(input, output, size, all_twiddles);
            }

            // Loop over remaining butterflies
            for iteration in 1..bfly.iterations {
                all_twiddles = &all_twiddles[(size / RADIX) * (RADIX - 1)..];
                size /= RADIX;
                if iteration % 2 == 0 {
                    unsafe {
                        bfly.butterfly
                            .apply_step::<Vector>(work, output, size, all_twiddles);
                    }
                } else {
                    unsafe {
                        bfly.butterfly
                            .apply_step::<Vector>(output, work, size, all_twiddles);
                    }
                }
            }

            (bfly.iterations % 2) != 0
        }
        safe::<T, Bfly, Vector, { RADIX }>(&self, input, output, work)
    }
    */

    #[inline(always)]
    unsafe fn apply_in_place<Vector>(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
    ) -> bool
    where
        Vector: crate::vector::ComplexVector<Float = T>,
    {
        #[inline(always)]
        fn safe<T, Bfly, Vector, const RADIX: usize>(
            bfly: &ButterflyStage<T, Bfly, { RADIX }>,
            input: &mut [Complex<T>],
            output: &mut [Complex<T>],
        ) -> bool
        where
            T: FftFloat,
            Bfly: Butterfly<T, { RADIX }>,
            Vector: crate::vector::ComplexVector<Float = T>,
        {
            assert_eq!(input.len(), output.len());
            assert!(bfly.iterations > 0);
            let mut size = bfly.size;
            let mut stride = input.len() / bfly.size;
            let mut all_twiddles: &[Complex<T>] = &bfly.twiddles;
            let mut iteration = 0;

            // Use partial loads until the stride is large enough
            while stride < Vector::WIDTH {
                let (from, to): (&mut _, &mut _) = if iteration % 2 == 0 {
                    (input, output)
                } else {
                    (output, input)
                };
                unsafe {
                    bfly.butterfly.apply_step_partial::<Vector>(
                        from,
                        to,
                        size,
                        stride,
                        all_twiddles,
                    );
                }
                size /= RADIX;
                stride *= RADIX;
                all_twiddles = &all_twiddles[size * (RADIX - 1)..];
                iteration += 1;
            }

            for iteration in iteration..bfly.iterations {
                let (from, to): (&mut _, &mut _) = if iteration % 2 == 0 {
                    (input, output)
                } else {
                    (output, input)
                };
                unsafe {
                    bfly.butterfly
                        .apply_step_full::<Vector>(from, to, size, stride, all_twiddles);
                }
                size /= RADIX;
                stride *= RADIX;
                all_twiddles = &all_twiddles[size * (RADIX - 1)..];
            }

            (bfly.iterations % 2) != 0
        }
        safe::<T, Bfly, Vector, { RADIX }>(&self, input, output)
    }
}

enum Stage<T: FftFloat> {
    Radix2(ButterflyStage<T, Radix2, 2>),
    Radix3(ButterflyStage<T, Radix3<T>, 3>),
    Radix4(ButterflyStage<T, Radix4, 4>),
    Radix8(ButterflyStage<T, Radix8<T>, 8>),
}

impl<T: FftFloat> Stage<T> {
    fn new(radix: usize, size: usize, iterations: usize, forward: bool) -> Self {
        if radix == 2 {
            return Self::Radix2(ButterflyStage::new(size, iterations, forward));
        }
        if radix == 3 {
            return Self::Radix3(ButterflyStage::new(size, iterations, forward));
        }
        if radix == 4 {
            return Self::Radix4(ButterflyStage::new(size, iterations, forward));
        }
        if radix == 8 {
            return Self::Radix8(ButterflyStage::new(size, iterations, forward));
        }
        unimplemented!("unsupported radix");
    }

    #[inline(always)]
    unsafe fn apply_in_place<V: ComplexVector<Float = T>>(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
    ) -> bool {
        match self {
            Self::Radix2(stage) => stage.apply_in_place::<V>(input, output),
            Self::Radix3(stage) => stage.apply_in_place::<V>(input, output),
            Self::Radix4(stage) => stage.apply_in_place::<V>(input, output),
            Self::Radix8(stage) => stage.apply_in_place::<V>(input, output),
        }
    }

    /*
    #[inline(always)]
    unsafe fn apply_out_of_place<V: ComplexVector<Float = T>>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        work: &mut [Complex<T>],
    ) -> bool {
        match self {
            Self::Radix2(stage) => stage.apply_out_of_place::<V>(input, output, work),
            Self::Radix3(stage) => stage.apply_out_of_place::<V>(input, output, work),
            Self::Radix4(stage) => stage.apply_out_of_place::<V>(input, output, work),
            Self::Radix8(stage) => stage.apply_out_of_place::<V>(input, output, work),
        }
    }
    */
}

fn get_stages<T: FftFloat>(mut size: usize) -> (Vec<Stage<T>>, Vec<Stage<T>>) {
    let mut forward_stages = Vec::new();
    let mut inverse_stages = Vec::new();
    /*
    {
        let (count, new_size) = num_factors(8, size);
        if count > 0 {
            forward_stages.push(Stage::new(8, size, count, true));
            inverse_stages.push(Stage::new(8, size, count, false));
            size = new_size;
        }
    }
    */
    {
        let (count, new_size) = num_factors(4, size);
        if count > 0 {
            forward_stages.push(Stage::new(4, size, count, true));
            inverse_stages.push(Stage::new(4, size, count, false));
            size = new_size;
        }
    }
    {
        let (count, new_size) = num_factors(3, size);
        if count > 0 {
            forward_stages.push(Stage::new(3, size, count, true));
            inverse_stages.push(Stage::new(3, size, count, false));
            size = new_size;
        }
    }
    {
        let (count, new_size) = num_factors(2, size);
        if count > 0 {
            forward_stages.push(Stage::new(2, size, count, true));
            inverse_stages.push(Stage::new(2, size, count, false));
            size = new_size;
        }
    }
    if size != 1 {
        unimplemented!("unsupported radix");
    }
    (forward_stages, inverse_stages)
}

#[inline(always)]
unsafe fn forward_in_place_impl<T, Vector>(
    stages: &Vec<Stage<T>>,
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
) where
    T: FftFloat,
    Vector: ComplexVector<Float = T>,
{
    let mut data_in_work = false;
    for stage in stages {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        data_in_work ^= stage.apply_in_place::<Vector>(from, to);
    }
    if data_in_work {
        input.copy_from_slice(&work);
    }
}

#[inline(always)]
unsafe fn inverse_in_place_impl<T, Vector>(
    stages: &Vec<Stage<T>>,
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
) where
    T: FftFloat,
    Vector: ComplexVector<Float = T>,
{
    let mut data_in_work = false;
    for stage in stages {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        data_in_work ^= stage.apply_in_place::<Vector>(from, to);
    }
    let scale = T::from_usize(input.len()).unwrap();
    if data_in_work {
        for (x, y) in work.iter().zip(input.iter_mut()) {
            *y = x / scale;
        }
    } else {
        for x in input.iter_mut() {
            *x /= scale;
        }
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn forward_in_place_f32_avx(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    unsafe {
        forward_in_place_impl::<f32, crate::vector::avx::Avx32>(stages, input, work);
    }
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => forward_in_place_f32_avx
)]
#[inline]
fn forward_in_place_f32(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    unsafe {
        forward_in_place_impl::<f32, crate::vector::generic::Generic<f32>>(stages, input, work);
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn inverse_in_place_f32_avx(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    unsafe {
        inverse_in_place_impl::<f32, crate::vector::avx::Avx32>(stages, input, work);
    }
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => inverse_in_place_f32_avx
)]
#[inline]
fn inverse_in_place_f32(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    unsafe {
        inverse_in_place_impl::<f32, crate::vector::generic::Generic<f32>>(stages, input, work);
    }
}

pub struct Fft32 {
    size: usize,
    forward_stages: Vec<Stage<f32>>,
    inverse_stages: Vec<Stage<f32>>,
    work: Box<[Complex<f32>]>,
}

impl Fft32 {
    pub fn new(size: usize) -> Self {
        let (forward_stages, inverse_stages) = get_stages(size);
        Self {
            size,
            forward_stages,
            inverse_stages,
            work: vec![Complex::default(); size].into_boxed_slice(),
        }
    }
}

impl Fft for Fft32 {
    type Float = f32;

    fn fft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        forward_in_place_f32(&self.forward_stages, input, &mut self.work);
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        inverse_in_place_f32(&self.inverse_stages, input, &mut self.work);
    }
}
