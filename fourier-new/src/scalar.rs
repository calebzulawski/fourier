use core::simd::SimdElement;

pub trait Scalar:
    SimdElement + Default + num_traits::FromPrimitive + num_traits::Float + num_traits::NumAssignOps
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}
