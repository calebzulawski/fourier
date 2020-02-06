#![allow(unused_macros)]

#[macro_export]
#[doc(hidden)]
macro_rules! generic_vector {
    { $type:ty } => {
        #[allow(unused_imports)]
        use num_complex::Complex;

        macro_rules! width {
            {} => { 1 }
        }

        macro_rules! zeroed {
            {} => { Complex::<$type>::default() }
        }

        macro_rules! broadcast {
            { $z:expr } => { $z }
        }

        macro_rules! add {
            { $a:expr, $b:expr } => { ($a + $b) }
        }

        macro_rules! sub {
            { $a:expr, $b:expr } => { ($a - $b) }
        }

        macro_rules! mul {
            { $a:expr, $b:expr } => { ($a * $b) }
        }

        macro_rules! rotate {
            { $z:expr, $positive:expr } => {
                {
                    if $positive {
                        Complex::<$type>::new(-$z.im, $z.re)
                    } else {
                        Complex::<$type>::new($z.im, -$z.re)
                    }
                }
            }
        }

        macro_rules! load_wide {
            { $from:expr } => { (*$from) }
        }

        macro_rules! store_wide {
            { $z:expr, $to:expr } => { { *$to = $z } }
        }

        macro_rules! load_narrow {
            { $from:expr } => { (*$from) }
        }

        macro_rules! store_narrow {
            { $z:expr, $to:expr } => { { *$to = $z } }
        }
    }
}
