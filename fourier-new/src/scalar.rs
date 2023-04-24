use core::{
    ops::{Add, Mul, Neg, Sub},
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

pub trait Scalar:
    SimdElement
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Default
    + num_traits::FromPrimitive
    + num_traits::Float
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}
