use num_complex::Complex;
use num_traits::{Float, FromPrimitive, NumAssign};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Normal;

fn dft<T: FromPrimitive + Float + NumAssign + Default + Clone>(
    input: &[Complex<T>],
    output: &mut [Complex<T>],
) {
    assert_eq!(input.len(), output.len());
    for k in 0..input.len() {
        output[k] = Complex::default();
        for n in 0..input.len() {
            let f = std::f64::consts::PI * ((2 * k * n) as f64) / (input.len() as f64);
            output[k] += input[n]
                * Complex::new(
                    T::from_f64(f.cos()).unwrap(),
                    T::from_f64(-f.sin()).unwrap(),
                );
        }
    }
}

fn idft<T: FromPrimitive + Float + NumAssign + Default + Clone>(
    input: &[Complex<T>],
    output: &mut [Complex<T>],
) {
    assert_eq!(input.len(), output.len());
    for k in 0..input.len() {
        output[k] = Complex::default();
        for n in 0..input.len() {
            let f = std::f64::consts::PI * ((2 * k * n) as f64) / (input.len() as f64);
            output[k] += input[n]
                * Complex::new(
                    T::from_f64(f.cos() / (input.len() as f64)).unwrap(),
                    T::from_f64(f.sin() / (input.len() as f64)).unwrap(),
                );
        }
    }
}

macro_rules! generate_dft_test {
    {
        $name:ident, $comparison:ident
    } => {
        #[test]
        fn $name() {
            let x = [
                Complex::new(0.07984231300862901, 0.2912053597430635),
                Complex::new(-0.3999645806965225, 0.5665336963535724),
                Complex::new(-1.0278505586819058, 0.503591759111203),
                Complex::new(-0.5847182112607883, 0.2852956847818571),
                Complex::new(0.8165939265478418, 0.48428811274975),
                Complex::new(-0.08194705182666534, 1.3634815124261457),
                Complex::new(-0.3447660142546443, -0.781105283625392),
                Complex::new(0.5282881452973941, -0.4680176663374855),
                Complex::new(-1.0689887834801322, 1.2245743551261743),
                Complex::new(-0.5118813091268151, -1.2811082751440426),
            ];

            let y = [
                Complex::new(-2.5953921244736087, 2.188739255184846),
                Complex::new(0.27239725684518834, -0.5487581762070741),
                Complex::new(1.2911356591694985, 0.4115497080289079),
                Complex::new(5.181762895312528, -4.330109311527908),
                Complex::new(1.432856335350818, 4.664992454671986),
                Complex::new(-0.4949461092468147, 1.2563693510247518),
                Complex::new(-2.0558954508390226, 1.2359845182788503),
                Complex::new(-0.7015751667411471, -0.6481366043854868),
                Complex::new(1.9167718867021326, -0.22783157531854403),
                Complex::new(-3.448692051993283, -1.0907460223196943),
            ];

            let mut dft_output = [Complex::default(); 10];
            let mut idft_output = [Complex::default(); 10];

            dft(&x, &mut dft_output);
            idft(&y, &mut idft_output);

            $comparison(&y, &dft_output);
            $comparison(&x, &idft_output);
        }
    }
}

generate_dft_test! { validate_dft_f32, near_f32 }
generate_dft_test! { validate_dft_f64, near_f64 }

fn near_f32(actual: &[Complex<f32>], expected: &[Complex<f32>]) {
    assert_eq!(actual.len(), expected.len());
    println!("actual: {:?}\nexpect: {:?}", actual, expected);
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
    println!("actual: {:?}\nexpect: {:?}", actual, expected);
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

macro_rules! generate_test {
    {
        $type:ty, $name:ident, $fft_gen:ident, $comparison:ident, $forward:expr
    } => {
        #[cfg(any(feature = "std", feature = "alloc"))]
        #[test]
        fn $name() {
            const MAX_SIZE: usize = 256;
            let stddev = if $forward {
                1.0
            } else {
                MAX_SIZE as $type
            };
            let distribution = Normal::new(0.0, stddev).unwrap();
            let rng: StdRng = SeedableRng::seed_from_u64(0xdeadbeef);
            let input = rng
                .sample_iter(&distribution)
                .zip(rand::thread_rng().sample_iter(&distribution))
                .take(MAX_SIZE)
                .map(|(x, y)| Complex::new(x, y))
                .collect::<Vec<_>>();
            let mut fft_output = vec![Complex::default(); MAX_SIZE];
            let mut dft_output = vec![Complex::default(); MAX_SIZE];
            let transform = if $forward {
                fourier::Transform::Fft
            } else {
                fourier::Transform::Ifft
            };
            let reference = if $forward {
                dft::<$type>
            } else {
                idft::<$type>
            };
            for size in 1..MAX_SIZE {
                println!("SIZE: {}", size);
                let fft = fourier::$fft_gen(size);
                println!("{:#?}", fft);
                fft.transform(&input[0..size], &mut fft_output[0..size], transform);
                reference(&input[0..size], &mut dft_output[0..size]);
                $comparison(&fft_output[0..size], &dft_output[0..size]);
            }
        }
    }
}

generate_test! { f32, integrity_forward_f32, create_fft_f32, near_f32, true }
generate_test! { f32, integrity_inverse_f32, create_fft_f32, near_f32, false }
generate_test! { f64, integrity_forward_f64, create_fft_f64, near_f64, true }
generate_test! { f64, integrity_inverse_f64, create_fft_f64, near_f64, false }

macro_rules! generate_static_test {
    {
        $type:ty, $fftname:ident, $name:ident, $comparison:ident, $forward:expr
    } => {
        #[test]
        fn $name() {
            use fourier::Fft;
            let fft = $fftname::default();
            let stddev = if $forward {
                1.0
            } else {
                fft.size() as $type
            };
            let distribution = Normal::new(0.0, stddev).unwrap();
            let rng: StdRng = SeedableRng::seed_from_u64(0xdeadbeef);
            let input = rng
                .sample_iter(&distribution)
                .zip(rand::thread_rng().sample_iter(&distribution))
                .take(fft.size())
                .map(|(x, y)| Complex::new(x, y))
                .collect::<Vec<_>>();
            let mut fft_output = vec![Complex::default(); fft.size()];
            let mut dft_output = vec![Complex::default(); fft.size()];
            let transform = if $forward {
                fourier::Transform::Fft
            } else {
                fourier::Transform::Ifft
            };
            let reference = if $forward {
                dft::<$type>
            } else {
                idft::<$type>
            };
            fft.transform(&input, &mut fft_output, transform);
            reference(&input, &mut dft_output);
            $comparison(&fft_output, &dft_output);
        }
    }
}

/*
#[fourier::static_fft(f32, 64)]
struct StaticFft64f32;
generate_static_test! { f32, StaticFft64f32, integrity_static_f32_64_forward, near_f32, true }
generate_static_test! { f32, StaticFft64f32, integrity_static_f32_64_inverse, near_f32, false }

#[fourier::static_fft(f64, 64)]
struct StaticFft64f64;
generate_static_test! { f64, StaticFft64f64, integrity_static_f64_64_forward, near_f64, true }
generate_static_test! { f64, StaticFft64f64, integrity_static_f64_64_inverse, near_f64, false }

#[fourier::static_fft(f32, 73)]
#[derive(Default)]
struct StaticFft73f32;
generate_static_test! { f32, StaticFft73f32, integrity_static_f32_73_forward, near_f32, true }
generate_static_test! { f32, StaticFft73f32, integrity_static_f32_73_inverse, near_f32, false }

#[fourier::static_fft(f64, 73)]
#[derive(Default)]
struct StaticFft73f64;
generate_static_test! { f64, StaticFft73f64, integrity_static_f64_73_forward, near_f64, true }
generate_static_test! { f64, StaticFft73f64, integrity_static_f64_73_inverse, near_f64, false }
*/
