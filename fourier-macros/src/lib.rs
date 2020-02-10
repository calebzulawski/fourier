//! This crate provides the procedural attribute macro for generating heapless, statically-sized
//! FFTs. This crate is reexported in the [`fourier`](../fourier/index.html) crate.

extern crate proc_macro;
use num_complex::Complex;
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Error, Ident, ItemStruct, LitInt, Result, Token,
};

struct Config {
    pub ty: Ident,
    pub comma_token2: Token![,],
    pub len: LitInt,
}

impl Parse for Config {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            ty: input.parse()?,
            comma_token2: input.parse()?,
            len: input.parse()?,
        })
    }
}

/// Implements a statically-sized heapless FFT for a struct.
///
/// The attribute takes two arguments, floating-point type (`f32` or `f64`) and a size.
/// This implementation does not require any heap allocations and is suitable for `#[no_std]`
/// usage.
///
/// The following implements `Fft<Real = f32>` for `StaticFft`:
/// ```
/// # use fourier_macros::static_fft;
/// #[static_fft(f32, 128)]
/// #[derive(Default)]
/// struct StaticFft;
/// ```
#[proc_macro_attribute]
pub fn static_fft(
    attr: proc_macro::TokenStream,
    source: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = parse_macro_input!(source as ItemStruct);
    let config = parse_macro_input!(attr as Config);
    create_static_fft(item, config)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn to_array_complex<T: ToTokens + Copy>(
    ty: &Ident,
    v: &[Complex<T>],
) -> (TokenStream, TokenStream) {
    let re = v.iter().map(|x| x.re);
    let im = v.iter().map(|x| x.im);
    let size = v.len();
    (
        quote! {
            [#(Complex::new(#re, #im)),*]
        },
        quote! {
            [Complex<#ty>; #size]
        },
    )
}

fn to_array<T: ToTokens + Copy>(ty: &Ident, v: &[T]) -> (TokenStream, TokenStream) {
    let size = v.len();
    (
        quote! {
            [#(#v),*]
        },
        quote! {
            [#ty; #size]
        },
    )
}

macro_rules! implement {
    {
        $type:ty, $name:ident
    } => {
        fn $name(item: ItemStruct, config: Config) -> Result<TokenStream> {
            type CplxVec = Vec<Complex<$type>>;
            type Autosort = fourier_algorithms::Autosort<$type, CplxVec, CplxVec>;
            type Bluesteins = fourier_algorithms::Bluesteins<$type, Autosort, CplxVec, CplxVec, CplxVec>;
            let ty: Ident = parse_quote! { $type };
            let size = config.len.base10_parse::<usize>()?;
            let name = item.ident.clone();
            let usize_ty: Ident = parse_quote!{ usize };
            if let Some(autosort) = Autosort::new(size) {
                let (forward_twiddles, twiddles_type) = to_array_complex(&ty, autosort.twiddles().0);
                let (inverse_twiddles, _) = to_array_complex(&ty, autosort.twiddles().1);
                let (counts, counts_type) = to_array(&usize_ty, &autosort.counts());
                let work_size = autosort.work_size();
                Ok(quote! {
                    #item

                    impl fourier_algorithms::Fft for #name {
                        type Real = #ty;

                        fn size(&self) -> usize {
                            #size
                        }

                        fn transform_in_place(
                            &self,
                            input: &mut [num_complex::Complex<Self::Real>],
                            transform: fourier_algorithms::Transform,
                        ) {
                            use num_complex::Complex;
                            use fourier_algorithms::Fft;
                            type WorkArray = [Complex<$type>; #work_size];

                            // Work around 32 element trait limit
                            struct Twiddles(#twiddles_type);
                            impl AsRef<[Complex<$type>]> for Twiddles {
                                fn as_ref(&self) -> &[Complex<$type>] {
                                    &self.0
                                }
                            }
                            struct Work(WorkArray);
                            impl AsMut<[Complex<$type>]> for Work {
                                fn as_mut(&mut self) -> &mut [Complex<$type>] {
                                    &mut self.0
                                }
                            }

                            const FORWARD_TWIDDLES: Twiddles = Twiddles(#forward_twiddles);
                            const INVERSE_TWIDDLES: Twiddles = Twiddles(#inverse_twiddles);
                            const COUNTS: #counts_type = #counts;
                            const WORK: Work = Work([Complex::<#ty>::new(0., 0.); #work_size]);
                            let autosort = unsafe {
                                fourier_algorithms::Autosort::<$type, Twiddles, Work>::new_from_parts(
                                    #size,
                                    COUNTS,
                                    FORWARD_TWIDDLES,
                                    INVERSE_TWIDDLES,
                                    WORK,
                                )
                            };
                            autosort.transform_in_place(input, transform);
                        }
                    }
                })
            } else {
                let bluesteins = Bluesteins::new(size);
                let (forward_w_twiddles, w_twiddles_type) = to_array_complex(&ty, bluesteins.w_twiddles().0);
                let (inverse_w_twiddles, _) = to_array_complex(&ty, bluesteins.w_twiddles().1);
                let (forward_x_twiddles, x_twiddles_type) = to_array_complex(&ty, bluesteins.x_twiddles().0);
                let (inverse_x_twiddles, _) = to_array_complex(&ty, bluesteins.x_twiddles().1);
                let work_size = bluesteins.work_size();
                let inner_fft_size = bluesteins.inner_fft_size();
                Ok(quote! {
                    #item

                    impl fourier_algorithms::Fft for #name {
                        type Real = #ty;

                        fn size(&self) -> usize {
                            #size
                        }

                        fn transform_in_place(
                            &self,
                            input: &mut [num_complex::Complex<Self::Real>],
                            transform: fourier_algorithms::Transform,
                        ) {
                            #[fourier::static_fft($type, #inner_fft_size)]
                            #[derive(Default)]
                            struct InnerFft;

                            use num_complex::Complex;
                            use fourier_algorithms::Fft;
                            type WorkArray = [Complex<$type>; #work_size];

                            // Work around 32 element trait limit
                            struct WTwiddles(#w_twiddles_type);
                            impl AsRef<[Complex<$type>]> for WTwiddles {
                                fn as_ref(&self) -> &[Complex<$type>] {
                                    &self.0
                                }
                            }
                            struct XTwiddles(#x_twiddles_type);
                            impl AsRef<[Complex<$type>]> for XTwiddles {
                                fn as_ref(&self) -> &[Complex<$type>] {
                                    &self.0
                                }
                            }
                            struct Work(WorkArray);
                            impl AsMut<[Complex<$type>]> for Work {
                                fn as_mut(&mut self) -> &mut [Complex<$type>] {
                                    &mut self.0
                                }
                            }

                            const FORWARD_W_TWIDDLES: WTwiddles = WTwiddles(#forward_w_twiddles);
                            const INVERSE_W_TWIDDLES: WTwiddles = WTwiddles(#inverse_w_twiddles);
                            const FORWARD_X_TWIDDLES: XTwiddles = XTwiddles(#forward_x_twiddles);
                            const INVERSE_X_TWIDDLES: XTwiddles = XTwiddles(#inverse_x_twiddles);
                            const WORK: Work = Work([Complex::<#ty>::new(0., 0.); #work_size]);
                            let bluesteins = unsafe {
                                fourier_algorithms::Bluesteins::<$type, InnerFft, WTwiddles, XTwiddles, Work>::new_from_parts(
                                    #size,
                                    InnerFft::default(),
                                    FORWARD_W_TWIDDLES,
                                    INVERSE_W_TWIDDLES,
                                    FORWARD_X_TWIDDLES,
                                    INVERSE_X_TWIDDLES,
                                    WORK,
                                )
                            };
                            bluesteins.transform_in_place(input, transform);
                        }
                    }
                })
            }
        }
    }
}
implement! { f32, create_static_fft_f32 }
implement! { f64, create_static_fft_f64 }

fn create_static_fft(item: ItemStruct, config: Config) -> Result<TokenStream> {
    let ident_f32: Ident = parse_quote! { f32 };
    let ident_f64: Ident = parse_quote! { f64 };
    if &config.ty == &ident_f32 {
        create_static_fft_f32(item, config)
    } else if &config.ty == &ident_f64 {
        create_static_fft_f64(item, config)
    } else {
        Err(Error::new(Span::call_site(), "expected f32 or f64"))
    }
}
