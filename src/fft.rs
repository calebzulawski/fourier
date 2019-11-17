use crate::operations;
//use multiversion::target_clones;
use num_complex::Complex;
use operations::{forward_f32_in_place, get_operations, inverse_f32_in_place, Operation};

pub struct Fft32 {
    size: usize,
    forward_ops: Vec<Operation<f32>>,
    inverse_ops: Vec<Operation<f32>>,
    work: Box<[Complex<f32>]>,
}

impl Fft32 {
    pub fn new(size: usize) -> Self {
        let (forward_ops, inverse_ops) = get_operations(size);
        Self {
            size,
            forward_ops,
            inverse_ops,
            work: vec![Complex::default(); size].into_boxed_slice(),
        }
    }

    pub fn fft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        forward_f32_in_place(&self.forward_ops, input, &mut self.work);
    }

    pub fn ifft_in_place(&mut self, input: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size, "input must match configured size");
        inverse_f32_in_place(&self.inverse_ops, input, &mut self.work);
    }
}
