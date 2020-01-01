use fourier::create_fft_f32;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
use serde::Deserialize;

fn near_f32(actual: &[Complex<f32>], expected: &[Complex<f32>]) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = f32::epsilon()
        * f32::from_usize(actual.len()).unwrap().log2()
        * f32::from_usize(15).unwrap();
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
            let mut fft = create_fft_f32(data.x.len());
            fft.ifft_in_place(&mut data.y);
            println!("{:?}\n{:?}", data.x, data.y);
            near_f32(&data.y, &data.x);
        }
    }
}

#[derive(Deserialize)]
struct Data<T: Clone + Float> {
    x: Vec<Complex<T>>,
    y: Vec<Complex<T>>,
}

std::include! {"vectors/generate_tests.rs"}
