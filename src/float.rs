use num_traits::{Float, FloatConst, FromPrimitive, NumAssign};

pub trait FftFloat: Float + FloatConst + FromPrimitive + NumAssign + Default + Clone {}
impl<T> FftFloat for T where T: Float + FloatConst + FromPrimitive + NumAssign + Default + Clone {}
