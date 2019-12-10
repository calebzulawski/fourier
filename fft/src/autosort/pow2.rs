use crate::autosort::Butterfly;
use crate::fft::Fft;
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use crate::vector::ComplexVector;
use num_complex::Complex;

fn make_twiddles<T: FftFloat, const RADIX: usize, const SIZE: usize, const FORWARD: bool>(
) -> Box<[Complex<T>]> {
    let mut twiddles = Vec::with_capacity(SIZE);
    for i in 1..RADIX {
        for j in 0..(SIZE / RADIX) {
            twiddles.push(compute_twiddle(i * j, SIZE, FORWARD));
        }
    }
    twiddles.into_boxed_slice()
}

#[inline(always)]
unsafe fn in_place_impl<
    T: FftFloat,
    Vector: ComplexVector<Float = T>,
    const FORWARD: bool,
    const SIZE: usize,
    const STRIDE: usize,
    const DATA_IN_WORK: bool,
>(
    input: &mut [Complex<T>],
    work: &mut [Complex<T>],
) {
    // Figure out where the data is
    let (from, to): (&mut _, &mut _) = if DATA_IN_WORK {
        (work, input)
    } else {
        (input, work)
    };

    // Apply butterfly
    if SIZE % 4 == 0 {
        let butterfly = crate::autosort::radix4::Radix4::create(FORWARD);
        let twiddles = make_twiddles::<T, 4, { SIZE }, { FORWARD }>();
        if STRIDE >= Vector::WIDTH {
            butterfly.apply_step_full::<Vector>(from, to, SIZE, STRIDE, &twiddles);
        } else {
            butterfly.apply_step_partial::<Vector>(from, to, SIZE, STRIDE, &twiddles);
        }
        if SIZE != 4 {
            return in_place_impl::<
                T,
                Vector,
                FORWARD,
                { SIZE / 4 },
                {
                    if STRIDE <= (std::usize::MAX / 4) {
                        STRIDE * 4
                    } else {
                        STRIDE
                    }
                },
                { !DATA_IN_WORK },
            >(input, work);
        }
    } else if SIZE % 2 == 0 {
        let butterfly = crate::autosort::radix2::Radix2 {};
        let twiddles = make_twiddles::<T, 2, { SIZE }, { FORWARD }>();
        if STRIDE >= Vector::WIDTH {
            butterfly.apply_step_full::<Vector>(from, to, SIZE, STRIDE, &twiddles);
        } else {
            butterfly.apply_step_partial::<Vector>(from, to, SIZE, STRIDE, &twiddles);
        }
    }

    // Finish by moving data and applying scaling as necessary
    let scale = T::from_usize(input.len()).unwrap();
    if FORWARD && !DATA_IN_WORK {
        input.copy_from_slice(&work);
    } else if !DATA_IN_WORK {
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
unsafe fn in_place_f32_avx<const FORWARD: bool, const SIZE: usize>(
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    in_place_impl::<f32, crate::vector::avx::Avx32, FORWARD, SIZE, 1, false>(input, work);
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => in_place_f32_avx
)]
#[inline]
fn in_place_f32<const FORWARD: bool, const SIZE: usize>(
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    unsafe {
        in_place_impl::<f32, crate::vector::generic::Generic<f32>, FORWARD, SIZE, 1, false>(
            input, work,
        );
    }
}

fn in_place_f32_dispatch<const FORWARD: bool, const SIZE: usize>(
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    assert_eq!(input.len(), work.len());
    if input.len() == SIZE {
        in_place_f32::<{ FORWARD }, { SIZE }>(input, work)
    } else {
        in_place_f32_dispatch::<{ FORWARD }, { SIZE / 2 }>(input, work)
    }
}

const MAX_POW_2: usize = 1 << 25;

pub struct PowerTwoFft32 {
    size: usize,
    work: Box<[Complex<f32>]>,
}

impl PowerTwoFft32 {
    pub fn new(size: usize) -> Option<Self> {
        if size.count_ones() == 1 && size <= MAX_POW_2 {
            Some(Self {
                size,
                work: vec![Complex::default(); size].into_boxed_slice(),
            })
        } else {
            None
        }
    }
}

impl Fft for PowerTwoFft32 {
    type Float = f32;

    fn fft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        in_place_f32_dispatch::<true, { MAX_POW_2 }>(input, &mut self.work);
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        in_place_f32_dispatch::<false, { MAX_POW_2 }>(input, &mut self.work);
    }
}
