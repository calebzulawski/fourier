use crate::float::FftFloat;
use num_complex::Complex;

mod radix2;
mod radix3;
use radix2::*;
use radix3::*;

fn compute_twiddle<T: FftFloat>(index: usize, size: usize, forward: bool) -> Complex<T> {
    let theta = (index * 2) as f64 * std::f64::consts::PI / size as f64;
    let twiddle = Complex::new(
        T::from_f64(theta.cos()).unwrap(),
        T::from_f64(-theta.sin()).unwrap(),
    );
    if forward {
        twiddle
    } else {
        twiddle.conj()
    }
}

struct BaseConfig<T> {
    twiddles: Vec<Complex<T>>,
    stride: usize,
    size: usize,
}

impl<T: FftFloat> BaseConfig<T> {
    fn new(size: usize, stride: usize, radix: usize, forward: bool) -> Self {
        assert_eq!(size % radix, 0);
        let m = size / radix;
        let mut twiddles = Vec::new();
        for i in 1..radix {
            for j in 0..m {
                twiddles.push(compute_twiddle(i * j, size, forward));
            }
        }
        Self {
            twiddles,
            stride,
            size,
        }
    }

    pub fn forward(size: usize, stride: usize, radix: usize) -> Self {
        Self::new(size, stride, radix, true)
    }

    pub fn inverse(size: usize, stride: usize, radix: usize) -> Self {
        Self::new(size, stride, radix, false)
    }
}

macro_rules! operations {
    {
        $([radix $radix:literal => $operation:ident, $f32_op:ident]),*
    } => {
        pub enum Operation<T: FftFloat> {
            $($operation($operation<T>)),*
        }

        pub fn get_operations<T: FftFloat>(size: usize) -> (Vec<Operation<T>>, Vec<Operation<T>>) {
            let mut forward_ops = Vec::new();
            let mut inverse_ops = Vec::new();
            let mut subsize = size;
            let mut stride = 1usize;
            while subsize != 1 {
                $(
                    if subsize % $radix == 0 {
                        forward_ops.push(Operation::$operation($operation::forward(subsize, stride)));
                        inverse_ops.push(Operation::$operation($operation::inverse(subsize, stride)));
                        subsize /= $radix;
                        stride *= $radix;
                        continue;
                    }
                )*
                unimplemented!("unsupported radix");
            }
            (forward_ops, inverse_ops)
        }

        //#[target_clones("x86_64+avx")]
        #[inline]
        fn apply_f32(operation: &Operation<f32>, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
            match operation {
                $(Operation::$operation(config) => $f32_op(input, output, config)),*
            }
        }
    }
}

operations! {
    [radix 3 => Radix3, radix3_f32],
    [radix 2 => Radix2, radix2_f32]
}

//#[target_clones("x86_64+avx")]
#[inline]
pub fn forward_f32_in_place(
    operations: &Vec<Operation<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    //#[static_dispatch]
    //use apply_f32;
    let mut data_in_work = false;
    for op in operations {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        apply_f32(op, from, to);
        data_in_work ^= true;
    }
    if data_in_work {
        input.copy_from_slice(&work);
    }
}

//#[target_clones("x86_64+avx")]
#[inline]
pub fn inverse_f32_in_place(
    operations: &Vec<Operation<f32>>,
    input: &mut [Complex<f32>],
    work: &mut [Complex<f32>],
) {
    //#[static_dispatch]
    //use apply_f32;
    let mut data_in_work = false;
    for op in operations {
        let (from, to): (&mut _, &mut _) = if data_in_work {
            (work, input)
        } else {
            (input, work)
        };
        apply_f32(op, from, to);
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
