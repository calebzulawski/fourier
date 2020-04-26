pub trait Array<T>: AsRef<[T]> + AsMut<[T]> {
    fn new(size: usize) -> Self;
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> Array<T> for Box<[T]>
where
    T: Default + Clone,
{
    fn new(size: usize) -> Self {
        vec![T::default(); size].into_boxed_slice()
    }
}
