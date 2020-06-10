use crate::float::Float;
use generic_simd::vector::{width, Complex, Handle, SizedHandle, Vector};
use num_complex as nc;

pub(crate) trait Butterfly<T, W, H>
where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
{
    type Buffer: AsRef<[H::Vector]> + AsMut<[H::Vector]>;

    fn radix() -> usize;

    fn make_buffer(handle: H) -> Self::Buffer;

    fn apply(handle: H, input: Self::Buffer, forward: bool) -> Self::Buffer;
}

pub(crate) struct Butterfly2;

impl<T, W, H> Butterfly<T, W, H> for Butterfly2
where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
    H::Vector: Complex,
{
    type Buffer = [H::Vector; 2];

    #[inline(always)]
    fn radix() -> usize {
        2
    }

    #[inline(always)]
    fn make_buffer(handle: H) -> Self::Buffer {
        [handle.zeroed(); 2]
    }

    #[inline(always)]
    fn apply(_: H, input: Self::Buffer, _: bool) -> Self::Buffer {
        [input[0] + input[1], input[0] - input[1]]
    }
}

pub(crate) struct Butterfly3;

impl<T, W, H> Butterfly<T, W, H> for Butterfly3
where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
    H::Vector: Complex,
{
    type Buffer = [H::Vector; 3];

    #[inline(always)]
    fn radix() -> usize {
        3
    }

    #[inline(always)]
    fn make_buffer(handle: H) -> Self::Buffer {
        [handle.zeroed(); 3]
    }

    #[inline(always)]
    fn apply(handle: H, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let t = crate::twiddle::compute_twiddle(1, 3, forward);
        let twiddle = handle.splat(t);
        let twiddle_conj = handle.splat(t.conj());
        [
            input[0] + input[1] + input[2],
            input[0] + input[1] * twiddle + input[2] * twiddle_conj,
            input[0] + input[1] * twiddle_conj + input[2] * twiddle,
        ]
    }
}

pub(crate) struct Butterfly4;

impl<T, W, H> Butterfly<T, W, H> for Butterfly4
where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
    H::Vector: Complex,
{
    type Buffer = [H::Vector; 4];

    #[inline(always)]
    fn radix() -> usize {
        4
    }

    #[inline(always)]
    fn make_buffer(handle: H) -> Self::Buffer {
        [handle.zeroed(); 4]
    }

    #[inline(always)]
    fn apply(handle: H, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let mut a = {
            let a0 = Butterfly2::apply(handle, [input[0], input[2]], forward);
            let a1 = Butterfly2::apply(handle, [input[1], input[3]], forward);
            [a0[0], a0[1], a1[0], a1[1]]
        };
        a[3] = if forward {
            a[3].mul_i()
        } else {
            a[3].mul_neg_i()
        };
        let b = {
            let b0 = Butterfly2::apply(handle, [a[0], a[2]], forward);
            let b1 = Butterfly2::apply(handle, [a[1], a[3]], forward);
            [b0[0], b0[1], b1[0], b1[1]]
        };
        [b[0], b[3], b[1], b[2]]
    }
}

pub(crate) struct Butterfly8;

impl<T, W, H> Butterfly<T, W, H> for Butterfly8
where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
    H::Vector: Complex,
{
    type Buffer = [H::Vector; 8];

    #[inline(always)]
    fn radix() -> usize {
        8
    }

    #[inline(always)]
    fn make_buffer(handle: H) -> Self::Buffer {
        [handle.zeroed(); 8]
    }

    #[inline(always)]
    fn apply(handle: H, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let t = crate::twiddle::compute_twiddle(1, 8, forward);
        let twiddle = handle.splat(t);
        let twiddle_neg = handle.splat(nc::Complex::new(-t.re, t.im));
        let a1 = Butterfly4::apply(handle, [input[0], input[2], input[4], input[6]], forward);
        let mut b1 = Butterfly4::apply(handle, [input[1], input[3], input[5], input[7]], forward);
        b1[1] = b1[1] * twiddle;
        b1[2] = if forward {
            b1[2].mul_neg_i()
        } else {
            b1[2].mul_i()
        };
        b1[3] = b1[3] * twiddle_neg;
        let a2 = Butterfly2::apply(handle, [a1[0], b1[0]], forward);
        let b2 = Butterfly2::apply(handle, [a1[1], b1[1]], forward);
        let c2 = Butterfly2::apply(handle, [a1[2], b1[2]], forward);
        let d2 = Butterfly2::apply(handle, [a1[3], b1[3]], forward);
        [a2[0], b2[0], c2[0], d2[0], a2[1], b2[1], c2[1], d2[1]]
    }
}

#[inline(always)]
pub(crate) fn apply_butterfly<T, H, B>(
    butterfly: B,
    handle: H,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    H: Handle<nc::Complex<T>>,
    B: Butterfly<T, width::W1, H>
        + Butterfly<T, width::W2, H>
        + Butterfly<T, width::W4, H>
        + Butterfly<T, width::W8, H>,
    <H as SizedHandle<nc::Complex<T>, width::W1>>::Vector: Complex,
    <H as SizedHandle<nc::Complex<T>, width::W2>>::Vector: Complex,
    <H as SizedHandle<nc::Complex<T>, width::W4>>::Vector: Complex,
    <H as SizedHandle<nc::Complex<T>, width::W8>>::Vector: Complex,
{
    if stride >= 8 {
        apply_butterfly_wide::<T, width::W8, H, B>(
            butterfly,
            handle,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else if stride >= 4 {
        apply_butterfly_wide::<T, width::W4, H, B>(
            butterfly,
            handle,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else if stride >= 2 {
        apply_butterfly_wide::<T, width::W2, H, B>(
            butterfly,
            handle,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else {
        apply_butterfly_narrow::<T, H, B>(
            butterfly,
            handle,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    }
}

#[inline(always)]
pub(crate) fn apply_butterfly_wide<T, W, H, B>(
    _butterfly: B,
    handle: H,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    W: width::Width,
    H: SizedHandle<nc::Complex<T>, W>,
    B: Butterfly<T, W, H>,
    H::Vector: Complex,
{
    let m = size / B::radix();

    assert_eq!(input.len(), size * stride);
    assert_eq!(output.len(), input.len());
    assert!(cached_twiddles.len() >= size);
    assert!(stride >= W::VALUE);

    let full_count = (stride - 1) / H::Vector::width() * H::Vector::width();
    let final_offset = stride - H::Vector::width();
    let input_vectors = handle.overlapping(input);
    for i in 0..m {
        let twiddles = {
            let mut twiddles = B::make_buffer(handle);
            for k in 1..B::radix() {
                twiddles.as_mut()[k] = handle
                    .splat(unsafe { cached_twiddles.as_ptr().add(i * B::radix() + k).read() });
            }
            twiddles
        };

        // Loop over full vectors, with a final overlapping vector
        for j in (0..full_count)
            .step_by(H::Vector::width())
            .chain(core::iter::once(final_offset))
        {
            // Load full vectors
            let mut scratch = B::make_buffer(handle);
            for k in 0..B::radix() {
                scratch.as_mut()[k] =
                    unsafe { input_vectors.get_unchecked(j + stride * (i + k * m)) };
            }

            // Butterfly with optional twiddles
            scratch = B::apply(handle, scratch, forward);
            if size != B::radix() {
                for k in 1..B::radix() {
                    scratch.as_mut()[k] *= twiddles.as_ref()[k];
                }
            }

            // Store full vectors
            let store = unsafe { output.as_mut_ptr().add(j + B::radix() * stride * i) };
            for k in 0..B::radix() {
                unsafe { scratch.as_ref()[k].write_ptr(store.add(stride * k)) };
            }
        }
    }
}

#[inline(always)]
pub(crate) fn apply_butterfly_narrow<T, H, B>(
    _butterfly: B,
    handle: H,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    H: SizedHandle<nc::Complex<T>, width::W1>,
    B: Butterfly<T, width::W1, H>,
    H::Vector: Complex,
{
    assert_eq!(input.len(), size * stride);
    assert_eq!(output.len(), input.len());
    assert!(cached_twiddles.len() >= size);

    let m = size / B::radix();
    for i in 0..m {
        let twiddles = {
            let mut twiddles = B::make_buffer(handle);
            for k in 1..B::radix() {
                twiddles.as_mut()[k] = handle
                    .splat(unsafe { cached_twiddles.as_ptr().add(i * B::radix() + k).read() });
            }
            twiddles
        };

        let load = unsafe { input.as_ptr().add(stride * i) };
        let store = unsafe { output.as_mut_ptr().add(B::radix() * stride * i) };
        for j in 0..stride {
            // Load a single value
            let mut scratch = B::make_buffer(handle);
            for k in 0..B::radix() {
                scratch.as_mut()[k] = handle.splat(unsafe { load.add(stride * k * m + j).read() });
            }

            // Butterfly with optional twiddles
            scratch = B::apply(handle, scratch, forward);
            if size != B::radix() {
                for k in 1..B::radix() {
                    scratch.as_mut()[k] *= twiddles.as_ref()[k];
                }
            }

            // Store a single value
            for k in 0..B::radix() {
                unsafe {
                    store
                        .add(stride * k + j)
                        .write(scratch.as_ref()[k].as_slice()[0])
                };
            }
        }
    }
}

macro_rules! implement {
    // the handle must be passed in due to something with macro hygiene
    {
        $handle:ident, $name32:ident, $name64:ident,$butterfly:ident
    } => {
        paste::item_with_macros! {
            implement! { @impl $handle, $name32, $butterfly, f32 }
            implement! { @impl $handle, $name64, $butterfly, f64 }
        }
    };
    {
        @impl $handle:ident, $name:ident, $butterfly:ident, $type:ty
    } => {
        #[generic_simd::dispatch($handle)]
        pub(crate) fn $name(
            input: &[nc::Complex<$type>],
            output: &mut [nc::Complex<$type>],
            size: usize,
            stride: usize,
            cached_twiddles: &[nc::Complex<$type>],
            forward: bool,
        ) {
            apply_butterfly(
                $butterfly,
                $handle,
                input,
                output,
                size,
                stride,
                cached_twiddles,
                forward,
            );
        }
    }
}

implement! { handle, radix2_f32, radix2_f64, Butterfly2 }
implement! { handle, radix3_f32, radix3_f64, Butterfly3 }
implement! { handle, radix4_f32, radix4_f64, Butterfly4 }
implement! { handle, radix8_f32, radix8_f64, Butterfly8 }
