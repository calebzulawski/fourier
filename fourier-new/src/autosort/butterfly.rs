use crate::scalar::Scalar;
use simd_complex::Complex;
use simd_traits::{num::Signed, Vector};

#[inline(always)]
pub(super) fn butterfly<T, V, const N: usize, const FORWARD: bool>(
    input: [Complex<V>; N],
) -> [Complex<V>; N]
where
    T: Scalar,
    V: Vector<Scalar = T> + Signed + Copy,
{
    let butterfly2 = butterfly::<T, V, 2, FORWARD>;
    let butterfly4 = butterfly::<T, V, 4, FORWARD>;

    match N {
        2 => [input[0] + input[1], input[0] - input[1]]
            .as_ref()
            .try_into()
            .unwrap(),
        3 => {
            let twiddle = Complex::splat(super::compute_twiddle::<T, FORWARD>(1, 3));
            let twiddle_conj = twiddle.conj();
            [
                input[0] + input[1] + input[2],
                input[0] + input[1] * twiddle + input[2] * twiddle_conj,
                input[0] + input[1] * twiddle_conj + input[2] * twiddle,
            ]
            .as_ref()
            .try_into()
            .unwrap()
        }
        4 => {
            let a = {
                let a0 = butterfly2([input[0], input[2]]);
                let a1 = butterfly2([input[1], input[3]]);
                [a0[0], a0[1], a1[0], a1[1]]
            };
            let b = {
                let b0 = butterfly2([a[0], a[2]]);
                let b1 = butterfly2([a[1], a[3]]);
                [b0[0], b0[1], b1[0], b1[1]]
            };
            [b[0], b[3], b[1], b[2]].as_ref().try_into().unwrap()
        }
        8 => {
            let rotate = |v: Complex<V>, positive| {
                if positive {
                    Complex {
                        re: -v.im,
                        im: v.re,
                    }
                } else {
                    Complex {
                        re: v.im,
                        im: -v.re,
                    }
                }
            };

            let twiddle = Complex::<V>::splat(super::compute_twiddle::<T, FORWARD>(1, 8));
            let twiddle_neg = Complex {
                re: -twiddle.re,
                im: twiddle.im,
            };
            let a1 = butterfly4([input[0], input[2], input[4], input[6]]);
            let mut b1 = butterfly4([input[1], input[3], input[5], input[7]]);
            b1[1] = b1[1] * twiddle;
            b1[2] = rotate(b1[2], !FORWARD);
            b1[3] = b1[3] * twiddle_neg;
            let a2 = butterfly2([a1[0], b1[0]]);
            let b2 = butterfly2([a1[1], b1[1]]);
            let c2 = butterfly2([a1[2], b1[2]]);
            let d2 = butterfly2([a1[3], b1[3]]);
            [a2[0], b2[0], c2[0], d2[0], a2[1], b2[1], c2[1], d2[1]]
                .as_ref()
                .try_into()
                .unwrap()
        }
        _ => unimplemented!(),
    }
}
