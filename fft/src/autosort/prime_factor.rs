#![allow(unused_unsafe)]
#![allow(unused_macros)]

use crate::fft::Fft;
use crate::float::FftFloat;
use crate::twiddle::compute_twiddle;
use num_complex::Complex;

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
    fn new(size: usize) -> Self {
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
            unimplemented!("unsupported radix");
        }
        Self {
            size,
            stages,
            forward_twiddles,
            reverse_twiddles,
        }
    }
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_4_avx_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 3 + i) };
    crate::stage!(
        narrow,
        4,
        butterfly4,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_4_avx_narrow
)]
fn radix_4_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 3 + i) };
    crate::stage!(
        narrow,
        4,
        butterfly4,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_3_avx_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 2 + i) };
    crate::stage!(
        narrow,
        3,
        butterfly3,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_3_avx_narrow
)]
fn radix_3_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 2 + i) };
    crate::stage!(
        narrow,
        3,
        butterfly3,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_2_avx_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    _forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j + i) };
    crate::stage!(
        narrow,
        2,
        butterfly2,
        input,
        output,
        _forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_2_avx_narrow
)]
fn radix_2_narrow(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    _forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j + i) };
    crate::stage!(
        narrow,
        2,
        butterfly2,
        input,
        output,
        _forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_4_avx_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 3 + i) };
    crate::stage!(
        wide,
        4,
        butterfly4,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_4_avx_wide
)]
fn radix_4_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 3 + i) };
    crate::stage!(
        wide,
        4,
        butterfly4,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_3_avx_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 2 + i) };
    crate::stage!(
        wide,
        3,
        butterfly3,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_3_avx_wide
)]
fn radix_3_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j * 2 + i) };
    crate::stage!(
        wide,
        3,
        butterfly3,
        input,
        output,
        forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn radix_2_avx_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    _forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::avx_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j + i) };
    crate::stage!(
        wide,
        2,
        butterfly2,
        input,
        output,
        _forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => radix_2_avx_wide
)]
fn radix_2_wide(
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
    _forward: bool,
    size: usize,
    stride: usize,
    twiddles: &[Complex<f32>],
) {
    crate::generic_vector! {};
    let get_twiddle = |i, j| unsafe { *twiddles.get_unchecked(j + i) };
    crate::stage!(
        wide,
        2,
        butterfly2,
        input,
        output,
        _forward,
        size,
        stride,
        get_twiddle
    );
}

#[multiversion::target("[x86|x86_64]+avx")]
unsafe fn width_avx() -> usize {
    4
}

#[multiversion::multiversion(
    "[x86|x86_64]+avx" => width_avx
)]
fn width() -> usize {
    1
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

pub struct PrimeFactorFft32 {
    stages: Stages<f32>,
    work: Box<[Complex<f32>]>,
}

impl PrimeFactorFft32 {
    pub fn new(size: usize) -> Self {
        Self {
            stages: Stages::new(size),
            work: vec![Complex::default(); size].into_boxed_slice(),
        }
    }
}

impl Fft for PrimeFactorFft32 {
    type Float = f32;

    fn fft_in_place(&mut self, input: &mut [Complex<f32>]) {
        apply_stage(input, &mut self.work, &self.stages, true);
    }

    fn ifft_in_place(&mut self, input: &mut [Complex<f32>]) {
        apply_stage(input, &mut self.work, &self.stages, false);
    }
}
