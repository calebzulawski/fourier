use num_traits::{Float, FloatConst, FromPrimitive, NumAssign};

/// Floating-point types used for performing fast Fourier transforms.
pub trait FftFloat: Float + FloatConst + FromPrimitive + NumAssign + Default + Clone {}
impl<T> FftFloat for T where T: Float + FloatConst + FromPrimitive + NumAssign + Default + Clone {}
