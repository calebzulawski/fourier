#[macro_export]
macro_rules! implement_generic {
    {
        $radix:literal,
        $input:expr,
        $output:expr,
        $config:expr,
        $butterfly:ident
    } => {
        {
            let BaseConfig {
                twiddles,
                stride,
                size,
                forward,
            } = $config;

            assert_eq!($input.len(), size * stride);
            assert_eq!($output.len(), size * stride);
            assert!(*stride != 0);

            if *size == $radix {
                for i in 0..*stride {
                    let mut scratch = [Complex::default(); $radix];
                    for k in 0..$radix {
                        scratch[k] = $input[i + k * stride];
                    }
                    scratch = $butterfly(scratch, *forward);
                    for k in 0..$radix {
                        $output[i + k * stride] = scratch[k];
                    }
                }
            } else {
                let m = size / $radix;
                for i in 0..m {
                    let mut wi = [Complex::default(); $radix - 1];
                    for k in 0..($radix - 1) {
                        wi[k] = twiddles[i + k * m];
                    }
                    for j in 0..*stride {
                        let mut scratch = [Complex::default(); $radix];
                        for k in 0..$radix {
                            scratch[k] = $input[j + stride * (i + k * m)];
                        }
                        scratch = $butterfly(scratch, *forward);
                        for k in 0..($radix - 1) {
                            scratch[k + 1] = scratch[k + 1] * wi[k];
                        }
                        for k in 0..$radix {
                            $output[j + stride * ($radix * i + k)] = scratch[k];
                        }
                    }
                }
            }
        }
    }
}
