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

trait Butterfly<T, const RADIX: usize> {
    fn new(forward: bool) -> Self;
    unsafe fn apply<Vector: ComplexVector<Float = T>>(
        &self,
        input: [Vector; RADIX],
    ) -> [Vector; RADIX];
}

struct ButterflyStage<T, Bfly, const RADIX: usize> {
    butterfly: Bfly,
    twiddles: Vec<Complex<T>>,
    size: usize,
    stride: usize,
}

impl<T, Bfly, const RADIX: usize> ButterflyStage<T, Bfly, { RADIX }>
where
    T: FftFloat,
    Bfly: Butterfly<T, { RADIX }>,
{
    fn new(size: usize, stride: usize, forward: bool) -> Self {
        assert_eq!(size % RADIX, 0);
        let m = size / RADIX;
        let mut twiddles = Vec::new();
        for i in 1..RADIX {
            for j in 0..m {
                twiddles.push(compute_twiddle(i * j, size, forward));
            }
        }
        Self {
            butterfly: Bfly::new(forward),
            twiddles,
            size,
            stride,
        }
    }

    #[inline(always)]
    unsafe fn apply<Vector>(&self, input: &[Complex<T>], output: &mut [Complex<T>])
    where
        Vector: crate::vector::ComplexVector<Float = T>,
        Bfly: Butterfly<T, { RADIX }>,
    {
        #[inline(always)]
        fn safe<T, Bfly, Vector, const RADIX: usize>(
            bfly: &ButterflyStage<T, Bfly, { RADIX }>,
            input: &[Complex<T>],
            output: &mut [Complex<T>],
        ) where
            Vector: crate::vector::ComplexVector<Float = T>,
            Bfly: Butterfly<T, { RADIX }>,
        {
            assert_eq!(input.len(), bfly.size * bfly.stride);
            assert_eq!(output.len(), bfly.size * bfly.stride);

            enum Method {
                FullWidth((usize, usize)),
                PartialWidth,
            }

            let method = if bfly.stride >= Vector::WIDTH {
                let full_count = ((bfly.stride - 1) / Vector::WIDTH) * Vector::WIDTH;
                let final_offset = bfly.stride - Vector::WIDTH;
                Method::FullWidth((full_count, final_offset))
            } else {
                Method::PartialWidth
            };

            let m = bfly.size / RADIX;
            for i in 0..m {
                // Load twiddle factors
                let twiddles = {
                    let mut twiddles = zeroed_array::<T, Vector, { RADIX }>();
                    for k in 1..RADIX {
                        let twiddle = &bfly.twiddles[i + (k - 1) * m];
                        twiddles[k] = unsafe { Vector::broadcast(&twiddle) };
                    }
                    twiddles
                };

                if let Method::FullWidth((full_count, final_offset)) = method {
                    // Loop over full vectors, with a final overlapping vector
                    for j in (0..full_count)
                        .step_by(Vector::WIDTH)
                        .chain(std::iter::once(final_offset))
                    {
                        // Load full vectors
                        let mut scratch = zeroed_array::<T, Vector, { RADIX }>();
                        let load = unsafe { input.as_ptr().add(j + bfly.stride * i) };
                        for k in 0..RADIX {
                            scratch[k] = unsafe { Vector::load(load.add(bfly.stride * k * m)) };
                        }

                        // Butterfly with optional twiddles
                        scratch = unsafe { bfly.butterfly.apply(scratch) };
                        if bfly.size != RADIX {
                            for k in 1..RADIX {
                                scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                            }
                        }

                        // Store full vectors
                        let store = unsafe { output.as_mut_ptr().add(j + RADIX * bfly.stride * i) };
                        for k in 0..RADIX {
                            unsafe { scratch[k].store(store.add(bfly.stride * k)) };
                        }
                    }
                } else {
                    // Load a partial vector
                    let mut scratch = zeroed_array::<T, Vector, { RADIX }>();
                    let load = unsafe { input.as_ptr().add(bfly.stride * i) };
                    for k in 0..RADIX {
                        scratch[k] = unsafe {
                            Vector::partial_load(load.add(bfly.stride * k * m), bfly.stride)
                        };
                    }

                    // Butterfly with optional twiddles
                    scratch = unsafe { bfly.butterfly.apply(scratch) };
                    if bfly.size != RADIX {
                        for k in 1..RADIX {
                            scratch[k] = unsafe { scratch[k].mul(&twiddles[k]) };
                        }
                    }

                    // Store a partial vector
                    let store = unsafe { output.as_mut_ptr().add(RADIX * bfly.stride * i) };
                    for k in 0..RADIX {
                        unsafe {
                            scratch[k].partial_store(store.add(bfly.stride * k), bfly.stride)
                        };
                    }
                }
            }
        }
        safe::<T, Bfly, Vector, { RADIX }>(&self, input, output);
    }
}

enum Stage<T: FftFloat> {
    Radix2(ButterflyStage<T, Radix2, 2>),
    Radix3(ButterflyStage<T, Radix3<T>, 3>),
    Radix4(ButterflyStage<T, Radix4, 4>),
    Radix8(ButterflyStage<T, Radix8<T>, 8>),
}

impl<T: FftFloat> Stage<T> {
    fn new(radix: usize, size: usize, stride: usize, forward: bool) -> Self {
        if radix == 2 {
            return Self::Radix2(ButterflyStage::new(size, stride, forward));
        }
        if radix == 3 {
            return Self::Radix3(ButterflyStage::new(size, stride, forward));
        }
        if radix == 4 {
            return Self::Radix4(ButterflyStage::new(size, stride, forward));
        }
        if radix == 8 {
            return Self::Radix8(ButterflyStage::new(size, stride, forward));
        }
        unimplemented!("unsupported radix");
    }

    #[inline(always)]
    unsafe fn apply<V: ComplexVector<Float = T>>(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) {
        match self {
            Self::Radix2(stage) => stage.apply::<V>(input, output),
            Self::Radix3(stage) => stage.apply::<V>(input, output),
            Self::Radix4(stage) => stage.apply::<V>(input, output),
            Self::Radix8(stage) => stage.apply::<V>(input, output),
        }
    }
}

fn get_stages<T: FftFloat>(size: usize) -> (Vec<Stage<T>>, Vec<Stage<T>>) {
    let mut forward_stages = Vec::new();
    let mut inverse_stages = Vec::new();
    let mut subsize = size;
    let mut stride = 1usize;
    if subsize % 4 == 0 {
        forward_stages.push(Stage::new(4, subsize, stride, true));
        inverse_stages.push(Stage::new(4, subsize, stride, false));
        subsize /= 4;
        stride *= 4;
    }
    while subsize != 1 {
        if subsize % 8 == 0 {
            forward_stages.push(Stage::new(8, subsize, stride, true));
            inverse_stages.push(Stage::new(8, subsize, stride, false));
            subsize /= 8;
            stride *= 8;
            continue;
        }
        if subsize % 4 == 0 {
            forward_stages.push(Stage::new(4, subsize, stride, true));
            inverse_stages.push(Stage::new(4, subsize, stride, false));
            subsize /= 4;
            stride *= 4;
            continue;
        }
        if subsize % 3 == 0 {
            forward_stages.push(Stage::new(3, subsize, stride, true));
            inverse_stages.push(Stage::new(3, subsize, stride, false));
            subsize /= 3;
            stride *= 3;
            continue;
        }
        if subsize % 2 == 0 {
            forward_stages.push(Stage::new(2, subsize, stride, true));
            inverse_stages.push(Stage::new(2, subsize, stride, false));
            subsize /= 2;
            stride *= 2;
            continue;
        }
        unimplemented!("unsupported radix");
    }
    (forward_stages, inverse_stages)
}

#[multiversion::target("[x86|x86_64]+avx")]
#[inline]
unsafe fn apply_stage_f32_avx(
    stage: &Stage<f32>,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    stage.apply::<crate::vector::avx::Avx32>(input, output);
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => apply_stage_f32_avx
)]
#[inline]
fn apply_stage_f32(stage: &Stage<f32>, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
    unsafe { stage.apply::<crate::vector::generic::Generic<f32>>(input, output) };
}

#[multiversion::target_clones("[x86|x86_64]+avx")]
#[inline]
fn forward_f32_in_place(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    #[static_dispatch]
    use apply_stage_f32;
    let mut data_in_work = false;
    for stage in stages {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        apply_stage_f32(stage, from, to);
        data_in_work ^= true;
    }
    if data_in_work {
        input.copy_from_slice(&work);
    }
}

#[multiversion::target_clones("[x86|x86_64]+avx")]
#[inline]
fn inverse_f32_in_place(
    stages: &Vec<Stage<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    #[static_dispatch]
    use apply_stage_f32;
    let mut data_in_work = false;
    for stage in stages {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        apply_stage_f32(stage, from, to);
        data_in_work ^= true;
    }
    let scale = input.len() as f32;
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
        forward_f32_in_place(&self.forward_stages, input, &mut self.work);
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        inverse_f32_in_place(&self.inverse_stages, input, &mut self.work);
    }
}
