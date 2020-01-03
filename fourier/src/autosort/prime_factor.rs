#![allow(unused_unsafe)]
#![allow(unused_macros)]

use crate::fft::Fft;
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use num_complex::Complex;
use std::cell::Cell;

fn num_factors(factor: usize, mut value: usize) -> (usize, usize) {
    let mut count = 0;
    while value % factor == 0 {
        value /= factor;
        count += 1;
    }
    (count, value)
}

fn extend_twiddles<T: FftFloat>(
    forward_twiddles: &mut Vec<Complex<T>>,
    reverse_twiddles: &mut Vec<Complex<T>>,
    size: usize,
    radix: usize,
    iterations: usize,
) {
    let mut subsize = size;
    for _ in 0..iterations {
        let m = subsize / radix;
        for i in 0..m {
            for j in 1..radix {
                forward_twiddles.push(compute_twiddle(i * j, subsize, true));
                reverse_twiddles.push(compute_twiddle(i * j, subsize, false));
            }
        }
        subsize /= radix;
    }
}

struct Stages<T> {
    size: usize,
    stages: Vec<(usize, usize)>,
    forward_twiddles: Vec<Complex<T>>,
    reverse_twiddles: Vec<Complex<T>>,
}

impl<T: FftFloat> Stages<T> {
    fn new(size: usize) -> Option<Self> {
        let mut current_size = size;
        let mut stages = Vec::new();
        let mut forward_twiddles = Vec::new();
        let mut reverse_twiddles = Vec::new();

        {
            let (count, new_size) = num_factors(4, current_size);
            if count > 0 {
                stages.push((4, count));
                extend_twiddles(
                    &mut forward_twiddles,
                    &mut reverse_twiddles,
                    current_size,
                    4,
                    count,
                );
            }
            current_size = new_size;
        }
        {
            let (count, new_size) = num_factors(4, current_size);
            if count > 0 {
                stages.push((4, count));
                extend_twiddles(
                    &mut forward_twiddles,
                    &mut reverse_twiddles,
                    current_size,
                    4,
                    count,
                );
            }
            current_size = new_size;
        }
        {
            let (count, new_size) = num_factors(3, current_size);
            if count > 0 {
                stages.push((3, count));
                extend_twiddles(
                    &mut forward_twiddles,
                    &mut reverse_twiddles,
                    current_size,
                    3,
                    count,
                );
            }
            current_size = new_size;
        }
        {
            let (count, new_size) = num_factors(2, current_size);
            if count > 0 {
                stages.push((2, count));
                extend_twiddles(
                    &mut forward_twiddles,
                    &mut reverse_twiddles,
                    current_size,
                    2,
                    count,
                );
            }
            current_size = new_size;
        }
        if current_size != 1 {
            None
        } else {
            Some(Self {
                size,
                stages,
                forward_twiddles,
                reverse_twiddles,
            })
        }
    }
}

macro_rules! make_radix_fns {
    {
        @impl $width:ident, $radix:literal, $name:ident, $butterfly:ident
    } => {
        #[multiversion::target_clones("[x86|x86_64]+avx")]
        fn $name(
            input: &[Complex<f32>],
            output: &mut [Complex<f32>],
            _forward: bool,
            size: usize,
            stride: usize,
            twiddles: &[Complex<f32>],
        ) {
            #[target_cfg(target = "[x86|x86_64]+avx")]
            crate::avx_vector! {};

            #[target_cfg(not(target = "[x86|x86_64]+avx"))]
            crate::generic_vector! {};

            let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * ($radix - 1) + i) };
            crate::stage!(
                $width,
                $radix,
                $butterfly,
                input,
                output,
                _forward,
                size,
                stride,
                get_twiddle
            );
        }
    };
    {
        $([$radix:literal, $wide_name:ident, $narrow_name:ident, $butterfly:ident]),*
    } => {
        $(
            make_radix_fns! { @impl wide, $radix, $wide_name, $butterfly }
            make_radix_fns! { @impl narrow, $radix, $narrow_name, $butterfly }
        )*
    };
}

make_radix_fns! {
    [2, radix_2_wide, radix_2_narrow, butterfly2],
    [3, radix_3_wide, radix_3_narrow, butterfly3],
    [4, radix_4_wide, radix_4_narrow, butterfly4]
}

#[multiversion::target_clones("[x86|x86_64]+avx")]
fn width() -> usize {
    #[target_cfg(target = "[x86|x86_64]+avx")]
    {
        4
    }

    #[target_cfg(not(target = "[x86|x86_64]+avx"))]
    {
        1
    }
}

#[multiversion::target_clones("[x86|x86_64]+avx")]
fn apply_stage(
    input: &mut [Complex<f32>],
    output: &mut [Complex<f32>],
    stages: &Stages<f32>,
    forward: bool,
) {
    #[static_dispatch]
    use radix_2_narrow;
    #[static_dispatch]
    use radix_2_wide;
    #[static_dispatch]
    use radix_3_narrow;
    #[static_dispatch]
    use radix_3_wide;
    #[static_dispatch]
    use radix_4_narrow;
    #[static_dispatch]
    use radix_4_wide;
    #[static_dispatch]
    use width;

    assert_eq!(input.len(), output.len());
    assert_eq!(stages.size, input.len());

    let width = width();

    let mut size = stages.size;
    let mut stride = 1;
    let mut twiddles: &[Complex<f32>] = if forward {
        &stages.forward_twiddles
    } else {
        &stages.reverse_twiddles
    };

    let mut data_in_output = false;
    for (radix, iterations) in &stages.stages {
        let mut iteration = 0;

        // Use partial loads until the stride is large enough
        while stride < width && iteration < *iterations {
            let (from, to): (&mut _, &mut _) = if data_in_output {
                (output, input)
            } else {
                (input, output)
            };
            match radix {
                4 => radix_4_narrow(from, to, forward, size, stride, twiddles),
                3 => radix_3_narrow(from, to, forward, size, stride, twiddles),
                2 => radix_2_narrow(from, to, forward, size, stride, twiddles),
                _ => unimplemented!("unsupported radix"),
            }
            size /= radix;
            stride *= radix;
            twiddles = &twiddles[size * (radix - 1)..];
            iteration += 1;
            data_in_output = !data_in_output;
        }

        for _ in iteration..*iterations {
            let (from, to): (&mut _, &mut _) = if data_in_output {
                (output, input)
            } else {
                (input, output)
            };
            match radix {
                4 => radix_4_wide(from, to, forward, size, stride, twiddles),
                3 => radix_3_wide(from, to, forward, size, stride, twiddles),
                2 => radix_2_wide(from, to, forward, size, stride, twiddles),
                _ => unimplemented!("unsupported radix"),
            }
            size /= radix;
            stride *= radix;
            twiddles = &twiddles[size * (radix - 1)..];
            data_in_output = !data_in_output;
        }
    }
    if forward {
        if data_in_output {
            input.copy_from_slice(output);
        }
    } else {
        let scale = stages.size as f32;
        if data_in_output {
            for (x, y) in output.iter().zip(input.iter_mut()) {
                *y = x / scale;
            }
        } else {
            for x in input.iter_mut() {
                *x /= scale;
            }
        }
    }
}

struct PrimeFactor32 {
    stages: Stages<f32>,
    work: Cell<Box<[Complex<f32>]>>,
    size: usize,
}

impl PrimeFactor32 {
    fn new(size: usize) -> Option<Self> {
        if let Some(stages) = Stages::new(size) {
            Some(Self {
                stages,
                work: Cell::new(vec![Complex::default(); size].into_boxed_slice()),
                size,
            })
        } else {
            None
        }
    }
}

impl Fft for PrimeFactor32 {
    type Float = f32;

    fn size(&self) -> usize {
        self.size
    }

    fn fft_in_place(&self, input: &mut [Complex<f32>]) {
        let mut work = self.work.take();
        apply_stage(input, &mut work, &self.stages, true);
        self.work.set(work);
    }

    fn ifft_in_place(&self, input: &mut [Complex<f32>]) {
        let mut work = self.work.take();
        apply_stage(input, &mut work, &self.stages, false);
        self.work.set(work);
    }
}

pub fn create_f32(size: usize) -> Option<Box<dyn Fft<Float = f32>>> {
    if let Some(fft) = PrimeFactor32::new(size) {
        Some(Box::new(fft))
    } else {
        None
    }
}
