use crate::array::Array;
use crate::fft::{Fft, Transform};
use crate::float::Float;
use core::cell::RefCell;
use core::marker::PhantomData;
use num_complex::Complex;

#[inline(always)]
fn split_radix<T: Float>(
    input: &[Complex<T>],
    output: &mut [Complex<T>],
    offset: usize,
    stride: usize,
    size: usize,
    forward: bool,
    max_size_mask: usize,
    twiddles: &[Complex<T>],
) {
    if size == 1 {
        output[0] = input[offset & max_size_mask];
    } else if size == 2 {
        let a = input[offset & max_size_mask];
        let b = input[offset.overflowing_add(stride).0 & max_size_mask];
        output[0] = a + b;
        output[1] = a - b;
    } else {
        split_radix(
            input,
            output,
            offset,
            stride * 2,
            size / 2,
            forward,
            max_size_mask,
            twiddles,
        );
        split_radix(
            input,
            &mut output[size / 2..],
            offset.overflowing_add(stride).0,
            stride * 4,
            size / 4,
            forward,
            max_size_mask,
            twiddles,
        );
        split_radix(
            input,
            &mut output[3 * size / 4..],
            offset.overflowing_sub(stride).0,
            stride * 4,
            size / 4,
            forward,
            max_size_mask,
            twiddles,
        );

        for k in 0..(size / 4) {
            let uk = output[k];
            let mut zk = output[k + size / 2];
            let uk2 = output[k + size / 4];
            let mut zdk = output[k + 3 * size / 4];

            let mut w = twiddles[k + size / 4];
            if !forward {
                w = w.conj();
            }
            zk *= w;
            zdk *= w.conj();

            let zp = zk + zdk;
            let mut zm = zk - zdk;
            zm = if forward {
                Complex::new(-zm.im, zm.re)
            } else {
                Complex::new(zm.im, -zm.re)
            };

            output[k] = uk + zp;
            output[k + size / 2] = uk - zp;
            output[k + size / 4] = uk2 - zm;
            output[k + 3 * size / 4] = uk2 + zm;
        }
    }
}

pub struct SplitRadix<T, Twiddles, Work> {
    size: usize,
    twiddles: Twiddles,
    work: RefCell<Work>,
    real_type: PhantomData<T>,
}

impl<T, Twiddles, Work> core::fmt::Debug for SplitRadix<T, Twiddles, Work> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("SplitRadix")
            .field("size", &self.size)
            .finish()
    }
}

impl<T, Twiddles, Work> SplitRadix<T, Twiddles, Work>
where
    T: Float,
    Twiddles: Array<Complex<T>>,
    Work: Array<Complex<T>>,
{
    pub fn new(size: usize) -> Option<Self> {
        if size.count_ones() != 1 {
            return None;
        };

        let mut twiddles = Twiddles::new(size * 4 / 2);
        let mut current_size = 1;
        while current_size <= size / 4 {
            for k in 0..current_size {
                let theta = -((k * 2) as f64) * core::f64::consts::PI / (current_size * 4) as f64;
                twiddles.as_mut()[current_size + k] = Complex::new(
                    T::from_f64(theta.cos()).unwrap(),
                    T::from_f64(theta.sin()).unwrap(),
                );
            }
            current_size *= 2;
        }

        let work = RefCell::new(Work::new(size));

        Some(Self {
            size,
            twiddles,
            work,
            real_type: PhantomData,
        })
    }
}

impl<T, Twiddles, Work> Fft for SplitRadix<T, Twiddles, Work>
where
    T: Float,
    Twiddles: Array<Complex<T>>,
    Work: Array<Complex<T>>,
{
    type Real = T;

    fn size(&self) -> usize {
        self.size
    }

    fn transform_in_place(&self, input: &mut [Complex<T>], transform: Transform) {
        assert_eq!(input.len(), self.size);

        // Obtain the work buffer
        let mut work_ref = self.work.borrow_mut();
        let work = work_ref.as_mut();

        // Apply
        split_radix(
            input,
            work,
            0,
            1,
            self.size,
            transform.is_forward(),
            self.size - 1,
            self.twiddles.as_ref(),
        );

        // Finish operation by scaling and moving data
        if let Some(scale) = match transform {
            Transform::Fft | Transform::UnscaledIfft => None,
            Transform::Ifft => Some(T::one() / T::from_usize(self.size).unwrap()),
            Transform::SqrtScaledFft | Transform::SqrtScaledIfft => {
                Some(T::one() / T::from_usize(self.size).unwrap().sqrt())
            }
        } {
            for (x, y) in work.iter().zip(input.iter_mut()) {
                *y = x * scale;
            }
        } else {
            input.copy_from_slice(work);
        }
    }
}

pub type HeapSplitRadix<T> = SplitRadix<T, Box<[Complex<T>]>, Box<[Complex<T>]>>;
