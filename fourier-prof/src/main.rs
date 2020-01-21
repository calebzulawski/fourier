use clap::{App, Arg};
use fourier::Fft;
use num_complex::Complex;
use rand::{distributions::Standard, Rng};

fn main() {
    let matches = App::new("fft-prof")
        .arg(Arg::with_name("size").takes_value(true).required(true))
        .get_matches();

    let size = usize::from_str_radix(matches.value_of("size").unwrap(), 10).unwrap();
    let fft = fourier::create_fft_f32(size);

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
