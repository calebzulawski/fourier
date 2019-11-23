use clap::{App, Arg};
use fft::Fft;
use num_complex::Complex;
use rand::{distributions::Standard, Rng};

fn main() {
    let matches = App::new("fft-prof")
        .arg(Arg::with_name("size").takes_value(true).required(true))
        .get_matches();

    let size = usize::from_str_radix(matches.value_of("size").unwrap(), 10).unwrap();
    let mut fft = fft::Fft32::new(size);

    let mut input = rand::thread_rng()
        .sample_iter(&Standard)
        .zip(rand::thread_rng().sample_iter(&Standard))
        .take(size)
        .map(|(x, y)| Complex::new(x, y))
        .collect::<Vec<_>>();

    loop {
        fft.fft_in_place(&mut input);
    }
}
