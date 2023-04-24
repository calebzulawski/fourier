pub trait Storage<T>: AsRef<[T]> + AsMut<[T]> {}

pub struct DynamicStorage<T>(Vec<T>);

impl<T> DynamicStorage<T>
where
    T: Default,
{
    pub fn new(count: usize) -> Self {
        Self(vec![T::Default(); count])
    }
}
