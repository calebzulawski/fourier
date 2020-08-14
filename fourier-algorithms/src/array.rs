#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{boxed::Box, vec};

/// An arraylike type.
pub trait Array<T>: AsRef<[T]> + AsMut<[T]> + Send + 'static {
    /// Construct an array with the given size.
    fn new(size: usize) -> Self;
}

macro_rules! box_impl {
    { $type:ty } => {
        #[cfg(any(feature = "std", feature = "alloc"))]
        impl Array<$type> for Box<[$type]> {
            fn new(size: usize) -> Self {
                generic_simd::alignment::allocate_max_aligned_slice(size)
            }
        }
    }
}

box_impl! { f32 }
box_impl! { f64 }
box_impl! { num_complex::Complex<f32> }
box_impl! { num_complex::Complex<f64> }

/// Creates a static array that implements [`Array`].
///
/// # Example
/// ```
/// use fourier_algorithms::{Array as _, make_array};
/// make_array! {
///     pub struct ArrayOf64([num_complex::Complex<f32>; 64]);
/// }
///
/// let array = ArrayOf64::new(64);
/// assert_eq!(array.as_ref().len(), 64);
/// ```
///
/// [`array`]: trait.Array.html
#[macro_export]
macro_rules! make_array {
    { $vis:vis struct $name:ident([$type:ty; $size:expr]); } => {
        $vis struct $name($crate::generic_simd::alignment::Aligned<$type, [$type; Self::SIZE]>);

        impl $name {
            const SIZE: usize = $size;
        }

        impl AsRef<[$type]> for $name {
            fn as_ref(&self) -> &[$type] {
                self.0.as_ref()
            }
        }

        impl AsMut<[$type]> for $name {
            fn as_mut(&mut self) -> &mut [$type] {
                self.0.as_mut()
            }
        }

        impl $crate::Array<$type> for $name {
            fn new(size: usize) -> Self {
                assert_eq!(size, Self::SIZE);
                Self($crate::generic_simd::alignment::Aligned::new([<$type>::default(); Self::SIZE]))
            }
        }
    }
}
