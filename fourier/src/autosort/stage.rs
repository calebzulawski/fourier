#[macro_export]
#[doc(hidden)]
macro_rules! stage {
    {
        wide, $radix:literal, $butterfly:ident, $input:ident, $output:ident, $forward: expr, $size:expr, $stride:expr, $stage_twiddles:expr
    } => {
        {
            let m = $size / $radix;

            let full_count = (($stride - 1) / width!()) * width!();
            let final_offset = $stride - width!();

            for i in 0..m {
                // Load twiddle factors
                let twiddles = {
                    let mut twiddles = [zeroed!(); $radix];
                    for k in 1..$radix {
                        let twiddle = $stage_twiddles(k, i);
                        twiddles[k] = broadcast!(twiddle);
                    }
                    twiddles
                };

                // Loop over full vectors, with a final overlapping vector
                for j in (0..full_count)
                    .step_by(width!())
                    .chain(std::iter::once(final_offset))
                {
                    // Load full vectors
                    let mut scratch = [zeroed!(); $radix];
                    let load = unsafe { $input.as_ptr().add(j + $stride * i) };
                    for k in 0..$radix {
                        scratch[k] = unsafe { load_wide!(load.add($stride * k * m)) };
                    }

                    // Butterfly with optional twiddles
                    scratch = $butterfly!(scratch, $forward);
                    if $size != $radix {
                        for k in 1..$radix {
                            scratch[k] = mul!(scratch[k], twiddles[k]);
                        }
                    }

                    // Store full vectors
                    let store = unsafe { $output.as_mut_ptr().add(j + $radix * $stride * i) };
                    for k in 0..$radix {
                        unsafe { store_wide!(scratch[k], store.add($stride * k)) };
                    }
                }
            }
        }
    };
    {
        narrow, $radix:literal, $butterfly:ident, $input:ident, $output:ident, $forward: expr, $size:expr, $stride:expr, $stage_twiddles:expr
    } => {
        {
            let m = $size / $radix;

            for i in 0..m {
                // Load twiddle factors
                let twiddles = {
                    let mut twiddles = [zeroed!(); $radix];
                    for k in 1..$radix {
                        let twiddle = $stage_twiddles(k, i);
                        twiddles[k] = broadcast!(twiddle);
                    }
                    twiddles
                };

                let load = unsafe { $input.as_ptr().add($stride * i) };
                let store = unsafe { $output.as_mut_ptr().add($radix * $stride * i) };
                for j in 0..$stride {
                    // Load a partial vector
                    let mut scratch = [zeroed!(); $radix];
                    for k in 0..$radix {
                        scratch[k] = unsafe { load_narrow!(load.add($stride * k * m + j)) };
                    }

                    // Butterfly with optional twiddles
                    scratch = $butterfly!(scratch, $forward);
                    if $size != $radix {
                        for k in 1..$radix {
                            scratch[k] = mul!(scratch[k], twiddles[k]);
                        }
                    }

                    // Store a partial vector
                    for k in 0..$radix {
                        unsafe { store_narrow!(scratch[k], store.add($stride * k + j)) };
                    }
                }
            }
        }
    }
}
