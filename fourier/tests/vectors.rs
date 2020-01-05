use fourier::{create_fft_f32, create_fft_f64};
use num_complex::Complex;
use num_traits::Float;
use serde::Deserialize;

fn near_f32(actual: &[Complex<f32>], expected: &[Complex<f32>]) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = 1e-4;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            float_cmp::approx_eq!(
                f32,
                actual.re,
                expected.re,
                float_cmp::F32Margin {
                    epsilon: tolerance,
                    ulps: 8
                }
            ) && float_cmp::approx_eq!(
                f32,
                actual.im,
                expected.im,
                float_cmp::F32Margin {
                    epsilon: tolerance,
                    ulps: 8
                }
            ),
            format!("{} != {}", actual, expected)
        );
    }
}

fn near_f64(actual: &[Complex<f64>], expected: &[Complex<f64>]) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = 1e-11;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            float_cmp::approx_eq!(
                f64,
                actual.re,
                expected.re,
                float_cmp::F64Margin {
                    epsilon: tolerance,
                    ulps: 8
                }
            ) && float_cmp::approx_eq!(
                f64,
                actual.im,
                expected.im,
                float_cmp::F64Margin {
                    epsilon: tolerance,
                    ulps: 8
                }
            ),
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
            let fft = create_fft_f32(data.x.len());
            fft.fft_in_place(&mut data.x);
            near_f32(&data.x, &data.y);
        }
    };
    {
        @inverse_f32 $test:ident, $file:tt
    } => {
        #[test]
        fn $test() {
            let serialized = std::include_str!($file);
            let mut data: Data<f32> = serde_json::from_str(serialized).unwrap();
            let fft = create_fft_f32(data.x.len());
            fft.ifft_in_place(&mut data.y);
            println!("{:?}\n{:?}", data.x, data.y);
            near_f32(&data.y, &data.x);
        }
    };
    {
        @forward_f64 $test:ident, $file:tt
    } => {
        #[test]
        fn $test() {
            let serialized = std::include_str!($file);
            let mut data: Data<f64> = serde_json::from_str(serialized).unwrap();
            let fft = create_fft_f64(data.x.len());
            fft.fft_in_place(&mut data.x);
            println!("{:?}\n{:?}", data.x, data.y);
            near_f64(&data.x, &data.y);
        }
    };
    {
        @inverse_f64 $test:ident, $file:tt
    } => {
        #[test]
        fn $test() {
            let serialized = std::include_str!($file);
            let mut data: Data<f64> = serde_json::from_str(serialized).unwrap();
            let fft = create_fft_f64(data.x.len());
            fft.ifft_in_place(&mut data.y);
            println!("{:?}\n{:?}", data.x, data.y);
            near_f64(&data.y, &data.x);
        }
    }
}

#[derive(Deserialize)]
struct Data<T: Clone + Float> {
    x: Vec<Complex<T>>,
    y: Vec<Complex<T>>,
}

std::include! {"vectors/generate_tests.rs"}
