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

use num_complex as nc;

/// Configuration for constructing FFTs.
#[derive(Debug, Clone)]
pub enum Configuration {
    Identity,
    Autosort(autosort::Configuration),
    Bluesteins(bluesteins::Configuration),
}

impl Configuration {
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
    BluesteinsXTwiddles,
    BluesteinsWTwiddles,
    BluesteinsWork,
> {
    Identity(identity::Identity<T>),
    Autosort(autosort::Autosort<T, AutosortTwiddles, AutosortWork>),
    Bluesteins(
        bluesteins::Bluesteins<
            T,
            autosort::Autosort<T, AutosortTwiddles, AutosortWork>,
            BluesteinsXTwiddles,
            BluesteinsWTwiddles,
            BluesteinsWork,
        >,
    ),
}

impl<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    > core::fmt::Debug
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
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
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    >
    Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
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
    pub fn to_boxed_fft(self) -> Box<dyn Fft<Real = T> + Send> {
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
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    > Fft
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
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
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    > Into<Box<dyn Fft<Real = T> + Send>>
    for Algorithm<
        T,
        AutosortTwiddles,
        AutosortWork,
        BluesteinsXTwiddles,
        BluesteinsWTwiddles,
        BluesteinsWork,
    >
where
    T: Float,
    AutosortTwiddles: Array<nc::Complex<T>>,
    AutosortWork: Array<nc::Complex<T>>,
    BluesteinsXTwiddles: Array<nc::Complex<T>>,
    BluesteinsWTwiddles: Array<nc::Complex<T>>,
    BluesteinsWork: Array<nc::Complex<T>>,
    autosort::Autosort<T, AutosortTwiddles, AutosortWork>: Fft<Real = T>,
{
    fn into(self) -> Box<dyn Fft<Real = T> + Send> {
        self.to_boxed_fft()
    }
}
