use crate::float::Float;
use generic_simd::{
    arch,
    vector::{
        scalar::{Scalar, ScalarWidth},
        slice::SliceWidth as _,
        width, Complex, SizedVector, Vector,
    },
};
use num_complex as nc;

pub(crate) trait Butterfly<T, W, Token>
where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
{
    type Buffer: AsRef<[SizedVector<nc::Complex<T>, W, Token>]>
        + AsMut<[SizedVector<nc::Complex<T>, W, Token>]>;

    fn radix() -> usize;

    fn make_buffer(token: Token) -> Self::Buffer;

    fn apply(token: Token, input: Self::Buffer, forward: bool) -> Self::Buffer;
}

pub(crate) struct Butterfly2;

impl<T, W, Token> Butterfly<T, W, Token> for Butterfly2
where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
    SizedVector<nc::Complex<T>, W, Token>: Complex,
{
    type Buffer = [SizedVector<nc::Complex<T>, W, Token>; 2];

    #[inline(always)]
    fn radix() -> usize {
        2
    }

    #[inline(always)]
    fn make_buffer(token: Token) -> Self::Buffer {
        [nc::Complex::<T>::zeroed(token); 2]
    }

    #[inline(always)]
    fn apply(_token: Token, input: Self::Buffer, _: bool) -> Self::Buffer {
        [input[0] + input[1], input[0] - input[1]]
    }
}

pub(crate) struct Butterfly3;

impl<T, W, Token> Butterfly<T, W, Token> for Butterfly3
where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
    SizedVector<nc::Complex<T>, W, Token>: Complex,
{
    type Buffer = [SizedVector<nc::Complex<T>, W, Token>; 3];

    #[inline(always)]
    fn radix() -> usize {
        3
    }

    #[inline(always)]
    fn make_buffer(token: Token) -> Self::Buffer {
        [nc::Complex::<T>::zeroed(token); 3]
    }

    #[inline(always)]
    fn apply(token: Token, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let t = crate::twiddle::compute_twiddle(1, 3, forward);
        let twiddle = t.splat(token);
        let twiddle_conj = t.conj().splat(token);
        [
            input[0] + input[1] + input[2],
            input[0] + input[1] * twiddle + input[2] * twiddle_conj,
            input[0] + input[1] * twiddle_conj + input[2] * twiddle,
        ]
    }
}

pub(crate) struct Butterfly4;

impl<T, W, Token> Butterfly<T, W, Token> for Butterfly4
where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
    SizedVector<nc::Complex<T>, W, Token>: Complex,
{
    type Buffer = [SizedVector<nc::Complex<T>, W, Token>; 4];

    #[inline(always)]
    fn radix() -> usize {
        4
    }

    #[inline(always)]
    fn make_buffer(token: Token) -> Self::Buffer {
        [nc::Complex::<T>::zeroed(token); 4]
    }

    #[inline(always)]
    fn apply(token: Token, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let mut a = {
            let a0 = Butterfly2::apply(token, [input[0], input[2]], forward);
            let a1 = Butterfly2::apply(token, [input[1], input[3]], forward);
            [a0[0], a0[1], a1[0], a1[1]]
        };
        a[3] = if forward {
            a[3].mul_i()
        } else {
            a[3].mul_neg_i()
        };
        let b = {
            let b0 = Butterfly2::apply(token, [a[0], a[2]], forward);
            let b1 = Butterfly2::apply(token, [a[1], a[3]], forward);
            [b0[0], b0[1], b1[0], b1[1]]
        };
        [b[0], b[3], b[1], b[2]]
    }
}

pub(crate) struct Butterfly8;

impl<T, W, Token> Butterfly<T, W, Token> for Butterfly8
where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
    SizedVector<nc::Complex<T>, W, Token>: Complex,
{
    type Buffer = [SizedVector<nc::Complex<T>, W, Token>; 8];

    #[inline(always)]
    fn radix() -> usize {
        8
    }

    #[inline(always)]
    fn make_buffer(token: Token) -> Self::Buffer {
        [nc::Complex::<T>::zeroed(token); 8]
    }

    #[inline(always)]
    fn apply(token: Token, input: Self::Buffer, forward: bool) -> Self::Buffer {
        let t = crate::twiddle::compute_twiddle(1, 8, forward);
        let twiddle = t.splat(token);
        let twiddle_neg = nc::Complex::new(-t.re, t.im).splat(token);
        let a1 = Butterfly4::apply(token, [input[0], input[2], input[4], input[6]], forward);
        let mut b1 = Butterfly4::apply(token, [input[1], input[3], input[5], input[7]], forward);
        b1[1] = b1[1] * twiddle;
        b1[2] = if forward {
            b1[2].mul_neg_i()
        } else {
            b1[2].mul_i()
        };
        b1[3] = b1[3] * twiddle_neg;
        let a2 = Butterfly2::apply(token, [a1[0], b1[0]], forward);
        let b2 = Butterfly2::apply(token, [a1[1], b1[1]], forward);
        let c2 = Butterfly2::apply(token, [a1[2], b1[2]], forward);
        let d2 = Butterfly2::apply(token, [a1[3], b1[3]], forward);
        [a2[0], b2[0], c2[0], d2[0], a2[1], b2[1], c2[1], d2[1]]
    }
}

#[inline(always)]
pub(crate) fn apply_butterfly<T, Token, B>(
    butterfly: B,
    token: Token,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    Token: arch::Token,
    B: Butterfly<T, width::W1, Token>
        + Butterfly<T, width::W2, Token>
        + Butterfly<T, width::W4, Token>
        + Butterfly<T, width::W8, Token>,
    nc::Complex<T>: Scalar<Token>,
    SizedVector<nc::Complex<T>, width::W1, Token>: Complex,
    SizedVector<nc::Complex<T>, width::W2, Token>: Complex,
    SizedVector<nc::Complex<T>, width::W4, Token>: Complex,
    SizedVector<nc::Complex<T>, width::W8, Token>: Complex,
{
    if stride >= 8 {
        apply_butterfly_wide::<T, width::W8, Token, B>(
            butterfly,
            token,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else if stride >= 4 {
        apply_butterfly_wide::<T, width::W4, Token, B>(
            butterfly,
            token,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else if stride >= 2 {
        apply_butterfly_wide::<T, width::W2, Token, B>(
            butterfly,
            token,
            input,
            output,
            size,
            stride,
            cached_twiddles,
            forward,
        );
    } else {
        apply_butterfly_narrow::<T, Token, B>(
            butterfly,
            token,
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
pub(crate) fn apply_butterfly_wide<T, W, Token, B>(
    _butterfly: B,
    token: Token,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    mut cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    W: width::Width,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, W>,
    B: Butterfly<T, W, Token>,
    SizedVector<nc::Complex<T>, W, Token>: Complex,
{
    let m = size / B::radix();

    assert_eq!(input.len(), size * stride);
    assert_eq!(output.len(), input.len());
    assert!(cached_twiddles.len() >= size);
    assert!(stride >= W::VALUE);

    let full_count = (stride - 1) / W::VALUE * W::VALUE;
    let final_offset = stride - W::VALUE;
    let input_vectors = input.overlapping(token);
    for i in 0..m {
        let twiddles = {
            let mut twiddles = B::make_buffer(token);
            for (v, t) in twiddles.as_mut().iter_mut().zip(cached_twiddles.iter()) {
                *v = t.splat(token);
            }
            twiddles
        };

        // Loop over full vectors, with a final overlapping vector
        for j in (0..full_count)
            .step_by(W::VALUE)
            .chain(core::iter::once(final_offset))
        {
            // Load full vectors
            let mut scratch = B::make_buffer(token);
            for k in 0..B::radix() {
                scratch.as_mut()[k] =
                    unsafe { input_vectors.get_unchecked(j + stride * (i + k * m)) };
            }

            // Butterfly with optional twiddles
            scratch = B::apply(token, scratch, forward);
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

        cached_twiddles = &cached_twiddles[B::radix()..];
    }
}

#[inline(always)]
pub(crate) fn apply_butterfly_narrow<T, Token, B>(
    _butterfly: B,
    token: Token,
    input: &[nc::Complex<T>],
    output: &mut [nc::Complex<T>],
    size: usize,
    stride: usize,
    mut cached_twiddles: &[nc::Complex<T>],
    forward: bool,
) where
    T: Float,
    Token: arch::Token,
    nc::Complex<T>: ScalarWidth<Token, width::W1>,
    B: Butterfly<T, width::W1, Token>,
    SizedVector<nc::Complex<T>, width::W1, Token>: Complex,
{
    assert_eq!(input.len(), size * stride);
    assert_eq!(output.len(), input.len());
    assert!(cached_twiddles.len() >= size);

    let m = size / B::radix();
    for i in 0..m {
        let twiddles = {
            let mut twiddles = B::make_buffer(token);
            for (v, t) in twiddles.as_mut().iter_mut().zip(cached_twiddles.iter()) {
                *v = t.splat(token);
            }
            twiddles
        };

        let load = unsafe { input.as_ptr().add(stride * i) };
        let store = unsafe { output.as_mut_ptr().add(B::radix() * stride * i) };
        for j in 0..stride {
            // Load a single value
            let mut scratch = B::make_buffer(token);
            for k in 0..B::radix() {
                scratch.as_mut()[k] = unsafe { load.add(stride * k * m + j).read() }.splat(token);
            }

            // Butterfly with optional twiddles
            scratch = B::apply(token, scratch, forward);
            if size != B::radix() {
                for k in 1..B::radix() {
                    scratch.as_mut()[k] *= twiddles.as_ref()[k];
                }
            }

            // Store a single value
            for k in 0..B::radix() {
                unsafe { store.add(stride * k + j).write(scratch.as_ref()[k][0]) };
            }
        }

        cached_twiddles = &cached_twiddles[B::radix()..];
    }
}

macro_rules! implement {
    // the token must be passed in due to something with macro hygiene
    {
        $token:ident, $name32:ident, $name64:ident,$butterfly:ident
    } => {
        paste::item_with_macros! {
            implement! { @impl $token, $name32, $butterfly, f32 }
            implement! { @impl $token, $name64, $butterfly, f64 }
        }
    };
    {
        @impl $token:ident, $name:ident, $butterfly:ident, $type:ty
    } => {
        #[generic_simd::dispatch($token)]
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
                $token,
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

implement! { token, radix2_f32, radix2_f64, Butterfly2 }
implement! { token, radix3_f32, radix3_f64, Butterfly3 }
implement! { token, radix4_f32, radix4_f64, Butterfly4 }
implement! { token, radix8_f32, radix8_f64, Butterfly8 }
