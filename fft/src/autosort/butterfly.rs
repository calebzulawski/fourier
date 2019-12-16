#[macro_export]
macro_rules! butterfly {
    { 2, $input:tt, $forward:tt } => { [add!($input[0], $input[1]), sub!($input[0], $input[1])] };
    { 3, $input:tt, $forward:tt } => {
        {
            let twiddle = broadcast!(crate::twiddle:::compute_twiddle(1, 3, forward));
            let twiddle_conj = broadcast!(&self.twiddle.conj());
            [
                add!($input[0], add!($input[1], $input[2])),
                add!($input[0], add!(mul!($input[1], twiddle), mul!($input[2], twiddle_conj))),
                add!($input[0], add!(mul!($input[1], twiddle_conj), mul!($input[2], twiddle))),
            ]
        }
    };
    { 4, $input:tt, $forward:tt } => {
        {
            let a1 = butterfly!(2, [x[0], x[2]]);
            let mut b1 = butterfly!(2, [x[1], x[3]]);
            b1[1] = rotate!(b1[1], $forward);
            let a2 = butterfly!(2, [a1[0], b1[0]]);
            let b2 = butterfly!(2, [a1[1], b1[1]]);
            [a2[0], b2[1], a2[1], b2[0]]
        }
    }
}
