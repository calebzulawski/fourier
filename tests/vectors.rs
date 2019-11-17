use fft::Fft32;
use num_complex::Complex;
use num_traits::Float;
use serde::Deserialize;

fn near(actual: &[Complex<f32>], expected: &[Complex<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).norm() < 1e-5,
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
            let mut fft = Fft32::new(data.x.len());
            fft.fft_in_place(&mut data.x);
            println!("{:?}", data.x);
            println!("{:?}", data.y);
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
            let mut fft = Fft32::new(data.x.len());
            fft.ifft_in_place(&mut data.y);
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
