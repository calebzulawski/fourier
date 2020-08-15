//! This crates provides `no_std` building blocks for performing fast Fourier transforms.  This
//! crate provides low level implementations with a low-level API, so you are probably looking for
//! the [`fourier`](../fourier/index.html) crate instead.
#![cfg_attr(not(feature = "std"), no_std)]

mod twiddle;

mod array;
pub mod autosort;
pub mod bluesteins;
mod fft;
mod float;
pub mod identity;

pub use array::*;
pub use fft::*;
pub use float::*;

#[doc(hidden)]
pub use generic_simd;

use num_complex as nc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::boxed::Box;

/// Configuration for constructing FFTs.
#[derive(Debug, Clone)]
pub enum Configuration {
    Identity,
    Autosort(autosort::Configuration),
    Bluesteins(bluesteins::Configuration),
}

impl Configuration {
    /// Construct a new configuration with the FFT size.
    pub const fn new(size: usize) -> Option<Self> {
        if size == 1 {
            Some(Self::Identity)
        } else if let Some(configuration) = autosort::Configuration::new(size) {
            Some(Self::Autosort(configuration))
        } else if let Some(configuration) = bluesteins::Configuration::new(size) {
            Some(Self::Bluesteins(configuration))
        } else {
            None
        }
    }
}

/// A dispatched FFT algorithm.
pub enum Algorithm<
    T,
    AutosortTwiddles,
    AutosortWork,
    BluesteinsWTwiddles,
    BluesteinsXTwiddles,
    BluesteinsWork,
> {
    Identity(identity::Identity<T>),
    Autosort(autosort::Autosort<T, AutosortTwiddles, AutosortWork>),
    Bluesteins(
        bluesteins::Bluesteins<
            T,
            autosort::Autosort<T, AutosortTwiddles, AutosortWork>,
            BluesteinsWTwiddles,
            BluesteinsXTwiddles,
            BluesteinsWork,
        >,
    ),
}

impl<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    > core::fmt::Debug
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    >
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        match self {
            Self::Identity(x) => x.fmt(f),
            Self::Autosort(x) => x.fmt(f),
            Self::Bluesteins(x) => x.fmt(f),
        }
    }
}

impl<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    >
    Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWork: Array<nc::Complex<T>>,
    autosort::Autosort<T, AutosortTwiddles, AutosortWork>: Fft<Real = T>,
{
    /// Creates a new FFT from a configuration.
    pub fn from_configuration(configuration: Configuration) -> Self {
        match configuration {
            Configuration::Identity => Self::Identity(identity::Identity::default()),
            Configuration::Autosort(configuration) => {
                Self::Autosort(autosort::Autosort::from_configuration(configuration))
            }
            Configuration::Bluesteins(configuration) => {
                Self::Bluesteins(bluesteins::Bluesteins::from_configuration(configuration))
            }
        }
    }

    /// Creates a new FFT.
    pub fn new(size: usize) -> Self {
        Self::from_configuration(Configuration::new(size).unwrap())
    }

    /// Creates a boxed FFT, skipping enum dispatch.
    #[cfg(any(feature = "std", feature = "alloc"))]
    pub fn into_boxed_fft(self) -> Box<dyn Fft<Real = T> + Send> {
        match self {
            Self::Identity(identity) => Box::new(identity),
            Self::Autosort(autosort) => Box::new(autosort),
            Self::Bluesteins(bluesteins) => Box::new(bluesteins),
        }
    }
}

impl<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    > Fft
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWork: Array<nc::Complex<T>>,
    autosort::Autosort<T, AutosortTwiddles, AutosortWork>: Fft<Real = T>,
{
    type Real = T;

    /// The size of the FFT.
    fn size(&self) -> usize {
        match self {
            Self::Identity(x) => x.size(),
            Self::Autosort(x) => x.size(),
            Self::Bluesteins(x) => x.size(),
        }
    }

    /// Apply an FFT or IFFT in-place.
    fn transform_in_place(&self, input: &mut [nc::Complex<Self::Real>], transform: Transform) {
        match self {
            Self::Identity(x) => x.transform_in_place(input, transform),
            Self::Autosort(x) => x.transform_in_place(input, transform),
            Self::Bluesteins(x) => x.transform_in_place(input, transform),
        }
    }

    /// Apply an FFT or IFFT out-of-place.
    fn transform(
        &self,
        input: &[nc::Complex<Self::Real>],
        output: &mut [nc::Complex<Self::Real>],
        transform: Transform,
    ) {
        match self {
            Self::Identity(x) => x.transform(input, output, transform),
            Self::Autosort(x) => x.transform(input, output, transform),
            Self::Bluesteins(x) => x.transform(input, output, transform),
        }
    }
}

/// A dispatched FFT algorithm allocated on the heap.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type HeapAlgorithm<T> = Algorithm<
    T,
    Box<[nc::Complex<T>]>,
    Box<[nc::Complex<T>]>,
    Box<[nc::Complex<T>]>,
    Box<[nc::Complex<T>]>,
    Box<[nc::Complex<T>]>,
>;

#[cfg(any(feature = "std", feature = "alloc"))]
impl<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    > Into<Box<dyn Fft<Real = T> + Send>>
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsWTwiddles,
        BluesteinsXTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWork: Array<nc::Complex<T>>,
    autosort::Autosort<T, AutosortTwiddles, AutosortWork>: Fft<Real = T>,
{
    fn into(self) -> Box<dyn Fft<Real = T> + Send> {
        self.into_boxed_fft()
    }
}

#[doc(hidden)]
pub use num_complex;

#[doc(hidden)]
pub use paste;

/// Create a stack-allocated FFT type.
///
/// The FFT type implements [`Fft`] and [`Default`].
/// The type does not use the heap and is suitable for `no_std` without allocation.
///
/// ```
/// # use fourier_algorithms::stack_fft;
/// stack_fft! {
///     pub struct Fft128 => f32, 128
/// }
/// ```
///
/// [`Fft`]: trait.Fft.html
/// [`Default`]: https://doc.rust-lang.org/std/default/trait.Default.html
#[macro_export]
macro_rules! stack_fft {
    { $vis:vis struct $name:ident => $real:ty, $size:literal } => {
        $crate::paste::paste! {
            // Array types
            $crate::make_array! {
                struct [<__AutosortTwiddles_ $name>]([$crate::num_complex::Complex<$real>; $name::AUTOSORT_TWIDDLES]);
            }
            $crate::make_array! {
                struct [<__AutosortSize_ $name>]([$crate::num_complex::Complex<$real>; $name::AUTOSORT_SIZE]);
            }
            $crate::make_array! {
                struct [<__FftSize_ $name>]([$crate::num_complex::Complex<$real>; $name::SIZE]);
            }

            // FFT
            $vis struct $name($crate::Algorithm<
                $real,
                [<__AutosortTwiddles_ $name>], // twiddles
                [<__AutosortSize_ $name>],     // work
                [<__AutosortSize_ $name>],     // Bluestein's w twiddles
                [<__FftSize_ $name>],          // Bluestein's x twiddles
                [<__AutosortSize_ $name>],     // Bluestein's inner FFT work
            >);

            impl core::default::Default for $name {
                fn default() -> Self {
                    Self($crate::Algorithm::from_configuration(Self::CONFIGURATION.unwrap()))
                }
            }

            impl core::fmt::Debug for $name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    self.0.fmt(f)
                }
            }

            impl $crate::Fft for $name {
                type Real = $real;

                #[inline]
                fn size(&self) -> usize {
                    use $crate::Fft as _;
                    self.0.size()
                }

                fn transform_in_place(
                    &self,
                    input: &mut [$crate::num_complex::Complex<Self::Real>],
                    transform: $crate::Transform
                ) {
                    use $crate::Fft as _;
                    self.0.transform_in_place(input, transform);
                }

                fn transform(
                    &self,
                    input: &[$crate::num_complex::Complex<Self::Real>],
                    output: &mut [$crate::num_complex::Complex<Self::Real>],
                    transform: $crate::Transform,
                ) {
                    use $crate::Fft as _;
                    self.0.transform(input, output, transform);
                }

            }

            impl $name {
                const SIZE: usize = $size;
                const CONFIGURATION: Option<$crate::Configuration> = $crate::Configuration::new(Self::SIZE);
                const AUTOSORT_TWIDDLES: usize = match Self::CONFIGURATION {
                    Some($crate::Configuration::Autosort(autosort)) => autosort.twiddles(),
                    Some($crate::Configuration::Bluesteins(bluesteins)) => bluesteins.inner_configuration().twiddles(),
                    _ => 0,
                };
                const AUTOSORT_SIZE: usize = match Self::CONFIGURATION {
                    Some($crate::Configuration::Bluesteins(bluesteins)) => bluesteins.inner_configuration().size(),
                    _ => Self::SIZE,
                };
            }
        }
    }
}
