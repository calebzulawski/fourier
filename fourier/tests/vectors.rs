use fourier::create_fft_f32;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
use serde::Deserialize;

fn near<T: Float + FromPrimitive + std::fmt::Display>(
    actual: &[Complex<T>],
    expected: &[Complex<T>],
) {
    assert_eq!(actual.len(), expected.len());
    let tolerance =
        T::epsilon() * T::from_usize(actual.len()).unwrap().log2() * T::from_usize(15).unwrap();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).norm() / expected.norm() < tolerance,
            format!("{} != {}", actual, expected)
        );
    }
}

macro_rules! generate_vector_test {
    {
        @forward_f32 $test:ident, $file:tt
    } => {
        #[test]
        fn $test() {
            let serialized = std::include_str!($file);
            let mut data: Data<f32> = serde_json::from_str(serialized).unwrap();
            let mut fft = create_fft_f32(data.x.len());
            fft.fft_in_place(&mut data.x);
            println!("{:?}\n{:?}", data.x, data.y);
            near(&data.x, &data.y);
        }
    };
    {
        @inverse_f32 $test:ident, $file:tt
    } => {
        #[test]
        fn $test() {
            let serialized = std::include_str!($file);
            let mut data: Data<f32> = serde_json::from_str(serialized).unwrap();
            let mut fft = create_fft_f32(data.x.len());
            fft.ifft_in_place(&mut data.y);
            println!("{:?}\n{:?}", data.x, data.y);
            near(&data.y, &data.x);
        }
    }
}

#[derive(Deserialize)]
struct Data<T: Clone + Float> {
    x: Vec<Complex<T>>,
    y: Vec<Complex<T>>,
}

std::include! {"vectors/generate_tests.rs"}
