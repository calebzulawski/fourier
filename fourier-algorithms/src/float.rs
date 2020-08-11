use num_traits as nt;

/// Floating-point types used for performing fast Fourier transforms.
pub trait Float:
    nt::Float + nt::FloatConst + nt::FromPrimitive + nt::NumAssign + Default + Clone + Send + 'static
{
}
impl<T> Float for T where
    T: nt::Float
        + nt::FloatConst
        + nt::FromPrimitive
        + nt::NumAssign
        + Default
        + Clone
        + Send
        + 'static
{
}
