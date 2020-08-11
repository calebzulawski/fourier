#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::{boxed::Box, vec};

/// An arraylike type
pub trait Array<T>: AsRef<[T]> + AsMut<[T]> + Send + 'static {
    fn new(size: usize) -> Self;
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> Array<T> for Box<[T]>
where
    T: Default + Clone + Send + 'static,
{
    fn new(size: usize) -> Self {
        vec![T::default(); size].into_boxed_slice()
    }
}
